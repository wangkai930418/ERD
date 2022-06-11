import numpy as np
import torch


class CategoriesSampler:
    def __init__(self, label, n_batch, n_cls, n_per,cls_per_task=5):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[: self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                rand_order = torch.randperm(len(l))

                pos = rand_order[: self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class Categories_Sampler_Exem_Distill_with_args:
    def __init__(self, label, n_batch, n_cls, n_shot, n_query, args, cls_per_task=20):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_shot + n_query
        self.n_shot = n_shot
        self.n_query = n_query
        self.cls_per_task = cls_per_task
        self.args = args

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        if self.args.probability == 1.0:
            for i_batch in range(self.n_batch):
                batch = []
                old_classes = torch.randperm(len(self.m_ind) - self.cls_per_task)[: self.n_cls]
                new_classes = torch.randperm(self.cls_per_task)[: self.n_cls] + len(self.m_ind) - self.cls_per_task
                for c in old_classes:
                    l = self.m_ind[c]
                    rand_order = torch.randperm(len(l))
                    if len(l) < self.n_per:
                        repeat_times = self.n_query // (len(l) - self.n_shot)
                        rand_order = torch.cat((rand_order, rand_order[self.n_shot:].repeat(repeat_times)))

                    pos = rand_order[: self.n_per]
                    batch.append(l[pos])

                for c in new_classes:
                    l = self.m_ind[c]
                    rand_order = torch.randperm(len(l))
                    if len(l) < self.n_per:
                        repeat_times = self.n_query // (len(l) - self.n_shot)
                        rand_order = torch.cat((rand_order, rand_order[self.n_shot:].repeat(repeat_times)))

                    pos = rand_order[: self.n_per]
                    batch.append(l[pos])

                batch = torch.stack(batch).t().reshape(-1)
                yield batch

        elif self.args.probability == 0:
            for i_batch in range(self.n_batch):
                batch = []
                old_classes = torch.randperm(len(self.m_ind) - self.cls_per_task)[: self.n_cls]
                new_classes = torch.randperm(len(self.m_ind))[: self.n_cls]

                for c in old_classes:
                    l = self.m_ind[c]
                    rand_order = torch.randperm(len(l))
                    if len(l) < self.n_per:
                        repeat_times = self.n_query // (len(l) - self.n_shot)
                        rand_order = torch.cat((rand_order, rand_order[self.n_shot:].repeat(repeat_times)))

                    pos = rand_order[: self.n_per]
                    batch.append(l[pos])

                for c in new_classes:
                    l = self.m_ind[c]
                    rand_order = torch.randperm(len(l))
                    if len(l) < self.n_per:
                        repeat_times = self.n_query // (len(l) - self.n_shot)
                        rand_order = torch.cat((rand_order, rand_order[self.n_shot:].repeat(repeat_times)))

                    pos = rand_order[: self.n_per]
                    batch.append(l[pos])

                batch = torch.stack(batch).t().reshape(-1)
                yield batch
        else:
            for i_batch in range(self.n_batch):
                batch = []
                if self.args.probability == 0.2:
                    inc = 1
                elif self.args.probability == 0.4:
                    inc = 2
                elif self.args.probability == 0.6:
                    inc = 3
                else:
                    print('not tested yet')
                    raise NotImplementedError

                if self.cls_per_task <= self.n_cls:
                    old_classes = torch.randperm(len(self.m_ind) - self.cls_per_task).repeat(2)[:self.n_cls + inc]
                else:
                    old_classes = torch.randperm(len(self.m_ind) - self.cls_per_task)[: self.n_cls + inc]

                new_classes = torch.randperm(self.cls_per_task)[: self.n_cls - inc] + len(
                    self.m_ind) - self.cls_per_task

                for c in old_classes:
                    l = self.m_ind[c]
                    rand_order = torch.randperm(len(l))
                    if len(l) < self.n_per:
                        repeat_times = self.n_query // (len(l) - self.n_shot)
                        rand_order = torch.cat((rand_order, rand_order[self.n_shot:].repeat(repeat_times)))

                    pos = rand_order[: self.n_per]
                    batch.append(l[pos])

                for c in new_classes:
                    l = self.m_ind[c]
                    rand_order = torch.randperm(len(l))

                    pos = rand_order[: self.n_per]
                    batch.append(l[pos])

                batch = torch.stack(batch).t().reshape(-1)
                yield batch


class Hard_Mine_Sampler:
    def __init__(self, label, class_id, n_batch, n_cls, n_shot, n_query):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_shot+n_query
        self.n_shot=n_shot
        self.n_query=n_query

        label = np.array(label)

        ind = np.argwhere(label == class_id).reshape(-1)
        ind = torch.from_numpy(ind)
        self.m_ind = ind


    def __len__(self):
        return self.n_batch

    def __iter__(self):
        batch = []
        batch.append(self.m_ind)
        batch = torch.stack(batch).t().reshape(-1)
        yield batch


