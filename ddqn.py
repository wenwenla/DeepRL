import torch
from dqn import DQNAgent


class DDQNAgent(DQNAgent):

    def update(self):
        s, a, r, s_, d = self.replay.sample(self.args.batch)
        with torch.no_grad():
            target = self.qnet(torch.Tensor(s))
            nxt_a = self.qnet(torch.Tensor(s_)).argmax(axis=1)
            nxt_q = self.target_qnet(torch.Tensor(s_))
            z = []
            for i, v in enumerate(nxt_a.numpy()):
                z.append(nxt_q[i, v])
            nxt_q = torch.Tensor(z)
            upd = self.args.gamma * nxt_q
            upd = torch.Tensor(r) + upd
            for i, v in enumerate(a):
                target[i, v] = r[i] if d[i] else upd[i]
        self.optim.zero_grad()
        q = self.qnet(torch.Tensor(s))
        loss = self.loss_fn(q, target)
        loss.backward()
        self.optim.step()
        self.soft_copy_parm()
        if self.steps % self.args.log_interval == 0:
            self.sw.add_scalar('loss/qloss', loss.item(), self.steps)

    def soft_copy_parm(self):
        with torch.no_grad():
            for t, s in zip(self.target_qnet.parameters(), self.qnet.parameters()):
                t.copy_(0.01 * t.data + 0.99 * s.data)
