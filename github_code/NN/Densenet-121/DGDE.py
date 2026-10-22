'''
DGDE 神经网络版：对应论文 Algorithm 1、3 及公式 (4)、(6)--(8)、(11)--(13)。
先生成试探参数，再执行一个 epoch 的 mini-batch Adam；小生境成员固定。
保留旧构造参数和 DGDE(optimizers, population, filename) 接口。
T_per 仅兼容旧接口；必须传入无 weight decay 的标准 Adam。
'''

import math
import random
from pathlib import Path

import joblib
import torch


class DGDE():
    def __init__(self, trainloader, testloader, valloader, criterion, lambda_reg,
                 population_size, niche_size, T_per, num_epoch, F, K_base, K_0,
                 pho_high, pho_low, fitness_p):
        for name, value, lower in [('population_size', population_size, 1),
                                   ('niche_size', niche_size, 1), ('num_epoch', num_epoch, 0)]:
            if int(value) != value or value < lower:
                raise ValueError(name + ' must be an integer >= ' + str(lower))
        if not math.isfinite(F) or F <= 0 or not 0 < K_base <= 1 or not math.isfinite(K_0) or K_0 <= 1:
            raise ValueError('Require F > 0, 0 < K_base <= 1 and K_0 > 1')
        if not all(0 <= p <= 1 for p in (pho_high, pho_low, fitness_p)):
            raise ValueError('pho_high, pho_low and fitness_p must be in [0, 1]')
        if lambda_reg != 0:
            raise ValueError('Equations (10)--(13) contain no L1 penalty; use lambda_reg=0')
        if getattr(criterion, 'reduction', 'mean') != 'mean':
            raise ValueError("The loss must use reduction='mean' for equations (10)--(11)")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.criterion = criterion
        self.lambda_reg = lambda_reg  # 不额外改变论文中的经验风险目标
        self.population_size = int(population_size)
        self.niche_size = int(niche_size)
        self.T_per = T_per  # Algorithm 3 每代重新分配策略，无周期性冻结
        self.num_epoch = int(num_epoch)
        self.F = F
        self.K_base = K_base
        self.K_0 = K_0
        self.pho_high = pho_high
        self.pho_low = pho_low
        self.fitness_p = fitness_p
        self.t = 0
        self._generation_batches = None
        self._reset_counts()

    def _reset_counts(self):
        self.counts = {'gradient_batches': 0, 'training_samples': 0,
                       'validation_evaluations': 0, 'validation_batches': 0,
                       'validation_samples': 0, 'test_evaluations': 0,
                       'test_batches': 0, 'test_samples': 0}

    def calculate_loss(self, model, inputs, targets):
        outputs = model(inputs)
        loss = self.criterion(outputs, targets)
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise ValueError('The mean loss must be a finite scalar')
        return loss, outputs

    def evaluate_model(self, model, dataloader, kind='validation'):
        # 按实际样本数加权，避免最后一个不足整批的小批次被过度加权。
        previous_mode = model.training
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        self.counts[kind + '_evaluations'] += 1
        try:
            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    loss, outputs = self.calculate_loss(model, inputs, targets)
                    size = targets.size(0)
                    total_loss += loss.item() * size
                    correct += outputs.argmax(dim=1).eq(targets).sum().item()
                    total += size
                    self.counts[kind + '_batches'] += 1
                    self.counts[kind + '_samples'] += size
        finally:
            model.train(previous_mode)
        if total == 0:
            raise ValueError(kind + ' loader must not be empty')
        return total_loss / total, 100.0 * correct / total

    def prob(self, cost, Fitness):
        spread = max(Fitness) - min(Fitness)
        return 0.0 if spread == 0 else (cost - min(Fitness)) / spread

    def _check_optimizer(self, model, optimizer):
        # 不能仅改变变异顺序而继续使用旧入口中的 SGD/AdamW。
        if type(optimizer) is not torch.optim.Adam:
            raise TypeError('Algorithm 3 requires torch.optim.Adam, not SGD or AdamW')
        params = [p for group in optimizer.param_groups for p in group['params']]
        expected = [p for p in model.parameters() if p.requires_grad]
        if len(params) != len(expected) or {id(p) for p in params} != {id(p) for p in expected}:
            raise ValueError('Each Adam must be bound to exactly its own model parameters')
        configs = []
        for group in optimizer.param_groups:
            if group.get('weight_decay', 0) != 0 or group.get('amsgrad', False) or group.get('maximize', False):
                raise ValueError('Use standard Adam: weight_decay=0, amsgrad=False, maximize=False')
            if group.get('decoupled_weight_decay', False):
                raise ValueError('Equation (13) does not use decoupled weight decay')
            configs.append((float(group['lr']), tuple(group['betas']), float(group['eps'])))
        if not configs or any(config != configs[0] for config in configs):
            raise ValueError('Equation (13) uses one lr/betas/eps configuration per individual')
        if configs[0][0] <= 0 or not math.isfinite(configs[0][0]) or configs[0][2] <= 0:
            raise ValueError('Adam learning rate and eps must be positive')
        return configs[0]

    def DGDE_update1(self, model, optimizer):
        """Algorithm 3 第 8--13 行：从当前试探参数开始 Adam 局部精修。"""
        self._check_optimizer(model, optimizer)
        # 原文 (12) 的偏差修正使用代内 b，而非跨代累计步数。
        # 原文漏写初值；这里明确补充 m^(t,0)=v^(t,0)=0（含最优个体）。
        optimizer.state.clear()
        model.train()
        batches = self.trainloader if self._generation_batches is None else self._generation_batches
        for inputs, targets in batches:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = self.calculate_loss(model, inputs, targets)
            loss.backward()
            optimizer.step()
            self.counts['gradient_batches'] += 1
            self.counts['training_samples'] += targets.size(0)
        return model

    @staticmethod
    def _parameters(model_or_snapshot):
        if isinstance(model_or_snapshot, torch.nn.Module):
            return tuple(p for p in model_or_snapshot.parameters() if p.requires_grad)
        return model_or_snapshot

    def DGDE_update2(self, model, best_model, optimizer):
        """先执行式 (6)，然后在试探参数处执行式 (11)--(13)。"""
        with torch.no_grad():
            for param, best_param in zip(self._parameters(model), self._parameters(best_model)):
                param.add_(self.K_base * (best_param.to(param.device) - param))
        return self.DGDE_update1(model, optimizer)

    def DGDE_update3(self, population, model, best_model, optimizer):
        """先执行式 (7)--(8)，再 Adam；population 是整代冻结的参数快照。"""
        j, k = random.sample(range(len(population)), 2)
        dynamic_K = self.K_0 + math.sqrt(self.t / self.num_epoch)
        # 先构造完整试探点再写入，避免独立调用本方法时 donor 与目标别名污染。
        trial = []
        with torch.no_grad():
            for param, best_param, pj, pk in zip(self._parameters(model), self._parameters(best_model),
                                                self._parameters(population[j]), self._parameters(population[k])):
                trial.append(param + dynamic_K * (best_param.to(param.device) - param)
                             + self.F * (pj.to(param.device) - pk.to(param.device)))
            for param, y in zip(self._parameters(model), trial):
                param.copy_(y)
        return self.DGDE_update1(model, optimizer)

    def model_distance(self, model1, model2):
        with torch.no_grad():
            return math.sqrt(sum(torch.sum((p1 - p2) ** 2).item()
                                 for p1, p2 in zip(self._parameters(model1), self._parameters(model2))))

    def distance_sort(self, individual, population):
        Distance = [self.model_distance(individual, p) for p in population]
        return sorted(zip(population, Distance, range(len(population))), key=lambda x: x[1])

    def fitness_sort(self, population):
        Fitness = [self.evaluate_model(m, self.valloader)[0] for m in population]
        Fitness_sort = sorted(zip(population, Fitness), key=lambda x: x[1])
        return Fitness_sort[0][0], Fitness, Fitness_sort

    def get_niche(self, population, Fitness=None):
        if Fitness is None:
            Fitness = [self.evaluate_model(m, self.valloader)[0] for m in population]
        Niche_Groups = []
        remaining_indices = list(range(len(population)))
        while remaining_indices:
            best_idx = min(remaining_indices, key=lambda i: Fitness[i])
            neighbors = [i for i in remaining_indices if i != best_idx]
            neighbors.sort(key=lambda i: self.model_distance(population[best_idx], population[i]))
            niche_indices = [best_idx] + neighbors[:self.niche_size - 1]
            Niche_Groups.append(niche_indices)
            assigned = set(niche_indices)
            remaining_indices = [i for i in remaining_indices if i not in assigned]
        return Niche_Groups

    def build_explore_set(self, niche_indices, population, fitness_sub):
        best_pos = min(range(len(fitness_sub)), key=lambda i: fitness_sub[i])
        E_i = []
        for pos, idx in enumerate(niche_indices):
            if pos == best_pos:
                continue
            p = self.prob(fitness_sub[pos], fitness_sub)
            # p=fitness_p 属于较差组；独立抽样，不固定探索人数。
            explore = random.random() >= self.pho_high if p < self.fitness_p else random.random() < self.pho_low
            if explore:
                E_i.append(idx)
        return E_i

    def weight_l1(self, model):
        # 仅作为诊断指标，未加到论文中的训练目标。
        with torch.no_grad():
            return sum(p.abs().sum() for p in model.parameters())

    def weight_l0(self, model, threshold):
        total = sum(p.numel() for p in model.parameters())
        active = sum((p.detach().abs() >= threshold).sum().item() for p in model.parameters())
        return active / total if total else 0.0  # 旧字段实际是非零比例，并非零元素比例

    def DGDE(self, optimizers, population, filename=None, evaluate_test=True):
        if len(population) != self.population_size or len(optimizers) != self.population_size:
            raise ValueError('population and optimizers must match population_size')
        if len({id(model) for model in population}) != self.population_size:
            raise ValueError('Population models must be independent objects')
        parameter_ids = [id(p) for model in population for p in model.parameters()]
        if len(set(parameter_ids)) != len(parameter_ids):
            raise ValueError('Population models must not share parameters')
        self.device = next(population[0].parameters()).device
        if any(p.device != self.device for m in population for p in m.parameters()):
            raise ValueError('All population parameters must be on the same device')
        configs = [self._check_optimizer(m, opt) for m, opt in zip(population, optimizers)]
        if any(config != configs[0] for config in configs):
            raise ValueError('Use the same Adam hyperparameters for every individual')
        self.t = 0
        self._reset_counts()
        self.niche_info_history = []
        thresholds = [0.01, 0.001, 0.0001]
        results = {'best_test_accs': [], 'best_test_losses': [],
                   'best_valid_accs': [], 'best_valid_losses': [],
                   'sparsity_epochs': {th: [] for th in thresholds},
                   'weight_epochs': [], 'num_niches': [], 'counts_history': []}
        metrics = [self.evaluate_model(m, self.valloader) for m in population]
        Fitness = [entry[0] for entry in metrics]
        Niche = self.get_niche(population, Fitness)  # Algorithm 1：只初始化一次
        self.Niche = [list(niche) for niche in Niche]

        while self.t < self.num_epoch:
            print(f'Epoch [{self.t + 1}/{self.num_epoch}]')
            # 整代只读 CPU 参数快照，避免后处理个体读取已更新的最优者或 donor。
            # 原模型对象保持不变，Adam 始终绑定正确的 Parameter；不额外复制 GPU 模型。
            parents = [tuple(p.detach().cpu().clone() for p in self._parameters(m)) for m in population]
            # 同一代所有个体使用相同的 mini-batches（包括随机增强后的张量）。
            # 缓存在 CPU；每代重新从 trainloader 生成，消耗一份训练数据的内存。
            self._generation_batches = [(x.detach().cpu().clone(), y.detach().cpu().clone())
                                        for x, y in self.trainloader]
            if not self._generation_batches:
                raise ValueError('trainloader must not be empty')
            niche_info = []
            try:
                for niche in Niche:
                    fitness_sub = [Fitness[idx] for idx in niche]
                    best_idx = niche[min(range(len(niche)), key=lambda pos: fitness_sub[pos])]
                    explore_set = set(self.build_explore_set(niche, parents, fitness_sub))
                    for idx in niche:
                        if idx == best_idx:
                            self.DGDE_update1(population[idx], optimizers[idx])
                        elif idx in explore_set:
                            self.DGDE_update3(parents, population[idx], parents[best_idx], optimizers[idx])
                        else:
                            self.DGDE_update2(population[idx], parents[best_idx], optimizers[idx])
                    niche_info.append({'indices': list(niche), 'best_idx': best_idx,
                                       'explore_set': sorted(explore_set)})
            finally:
                self._generation_batches = None
            del parents
            self.t += 1
            self.niche_info_history.append(niche_info)
            metrics = [self.evaluate_model(m, self.valloader) for m in population]
            Fitness = [entry[0] for entry in metrics]
            best_idx = min(range(len(population)), key=lambda idx: Fitness[idx])
            best_ind = population[best_idx]
            v_loss, v_acc = metrics[best_idx]
            results['best_valid_losses'].append(v_loss)
            results['best_valid_accs'].append(v_acc)
            # 第 5.2 节：测试集只用于最终泛化评价，不参与策略或个体选择。
            results['best_test_losses'].append(float('nan'))
            results['best_test_accs'].append(float('nan'))
            results['weight_epochs'].append(self.weight_l1(best_ind).item())
            results['num_niches'].append(len(Niche))
            for th in thresholds:
                results['sparsity_epochs'][th].append(self.weight_l0(best_ind, th))
            results['counts_history'].append(dict(self.counts))
            print(f'Best Valid Loss: {v_loss:.6f}, Best Valid Acc: {v_acc:.2f}%')

        best_idx = min(range(len(population)), key=lambda idx: Fitness[idx])
        self.pop = population
        self.g_best = population[best_idx]  # 最终种群中验证损失最小的网络
        t_loss, t_acc = (self.evaluate_model(self.g_best, self.testloader, kind='test')
                         if evaluate_test else (float('nan'), float('nan')))
        if self.t:
            results['best_test_losses'][-1] = t_loss
            results['best_test_accs'][-1] = t_acc
        results['final'] = {'best_idx': best_idx, 'valid_loss': metrics[best_idx][0],
                            'valid_acc': metrics[best_idx][1], 'test_loss': t_loss, 'test_acc': t_acc}
        results['counts'] = dict(self.counts)
        results['algorithm'] = {'version': 'DGDE-paper-v2', 'optimizer': 'Adam',
                                'adam_state': 'reset_each_generation', 'niches': 'fixed',
                                'donors': 'global_distinct', 'test_policy': 'final_only' if evaluate_test else 'disabled_during_tuning',
                                'batch_policy': 'shared_per_generation', 'lr': configs[0][0],
                                'betas': configs[0][1], 'eps': configs[0][2]}
        results['niche_info_history'] = self.niche_info_history
        results['initial_niches'] = self.Niche
        self.results = results
        if filename is not None:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, 'wb') as f:
                joblib.dump(results, f)
        return results
