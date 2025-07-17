import copy
import torch.nn.functional as F

import torch
import torch.nn as nn

class MemoryModule(nn.Module):
    """
    Using MLP as long term memeory, nothing fancy
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class MAC(nn.Module):
    """
    Mac arcgutecture
    """
    def __init__(self, d_model, num_heads, num_persistent, segment_size, memory_hidden_dim, memory_num_layers):
        super().__init__()
        self.d_model = d_model
        self.segment_size = segment_size
        # persistent memory does not change, just gets concat
        self.persistent_memory = nn.Parameter(torch.randn(num_persistent, d_model))
        # Long term memory Module
        self.memory = MemoryModule(d_model, memory_hidden_dim, d_model, memory_num_layers)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        # short term attention mechanism

        self.output_projection = nn.Linear(d_model, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # optmizer with momentum adn decay as mentioned in pape, this is an indredt implementation but in
        # interest of time I used this  SGD # this is used in test and train time
        self.memory_optimizer = torch.optim.SGD(self.memory.parameters(), lr=0.1, momentum=0.9, weight_decay=0.01)
        # For training purposes the paper mentions to use a optimizer for training. This would not be used in test time where this paper is excelling
        self.model_optimizer = torch.optim.AdamW(self.parameters(), lr=4e-4)

    def forward(self, x, target = None,test = False):
        """

        :param x:
        :param target: would be just present in training task I guess, not very sure tho, so while inference we wont have target
        :return:
        """
        batch_size, seq_len, _ = x.shape
        num_segments = (seq_len + self.segment_size - 1) // self.segment_size
        outputs = []
        # I am copying the as during test time my memory gets updated using write operations
        # now when new exemplar comes in, I want it to run on old memory
        # so I have stored it insomething like a deep copy which replaces the old Memory moel with new one
        # Ideally I should save the weights but I wanted a lazy way due to interest of time
        # implemeted this in eval pipeline
        # memory_clone = copy.deepcopy(self.memory)#
        # self.memory = memory_clone
        # S-NIAH results suggest that
        # Chunking
        # Here I am processing chunks as sequence for the sake of simplicity. But paper states that
        # inner loops can be writtten informed of matmul operations
        memory_loss = 0
        for t in range(num_segments):
            """
            Here I have just implemented memory loss not task loss as that is part of traning piplien"""
            start = t * self.segment_size
            end = min(start + self.segment_size, seq_len)
            # getting a segment
            S_t = x[:, start:end, :]
            q_t = self.W_Q(S_t)
            h_t = self.memory(q_t)
            # get persistent memory
            pers_mem = self.persistent_memory.unsqueeze(0).repeat(batch_size, 1, 1)
            # concat with persistent memory
            concat_seq = torch.cat([pers_mem, h_t, S_t], dim=1)
            seq_len_concat = concat_seq.shape[1]
            # upper trigangular matrix, wont go in much details with attention mechanism
            mask = torch.triu(torch.ones(seq_len_concat, seq_len_concat), diagonal=1).bool().to(x.device)
            y_t, _ = self.attention(concat_seq, concat_seq, concat_seq, attn_mask=mask)
            y_t = y_t[:, -self.segment_size:, :]
            # Combine attention output with memory output (approximating o_t = y_t x M_t(y_t)) || Usure what this operation is, is it a convolution or
            # just a element multiplication? , for now I am implemented a element wise multiplication
            m_t = self.memory(y_t)
            y_t = self.output_projection(y_t * m_t)  # Element-wise multiplication
            outputs.append(y_t)
            k_t = self.W_K(S_t)
            v_t = self.W_V(S_t)
            pred_v_t = self.memory(k_t)
            loss = ((pred_v_t - v_t) ** 2).mean()
            memory_loss += loss
            # earlier I implemented .backward in forward which can result into disasters. I dont want to hide my mistakes and thus just commented out
            # Figured you would love to know my though process more than the working of the actual Architecture
            """
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            # if test:
            #     self.memory_optimizer.zero_grad()
            #     loss.backward(retain_graph=True)
            #     self.memory_optimizer.step()
            """

        # reseting memory to ensure no cross interference occur while testing so this never get updated
        # if test:
        #     self.memory = memory_clone
        memory_loss = memory_loss / num_segments

        output = torch.cat(outputs, dim=1)
        task_loss = None
        if target is not None:
            task_loss = F.mse_loss(output, target, reduction='mean')

        return output, memory_loss, task_loss


    def process_sequence(self, x, target=None, test = False):
        # Process sequence
        # ideally I will have a state dict stored here, but not right now as I did lazy implementation
        output, memory_loss, task_loss = self.forward(x, target,test = test)

        return output, memory_loss, task_loss

    def update_memory(self, x):
        """
        Perform memory update step with gradient backward and optimizer step.
        Called during test time adaptation.
        # this would be only called in testing i think but can be called in training , not very sure
        This just updates the memory i think as this just work on memeory losses not task losses, however,
        the model_optimizer works on combination of both.
        """
        self.memory_optimizer.zero_grad()
        output, memory_loss, _ = self.forward(x, test=True)
        memory_loss.backward()
        self.memory_optimizer.step()
        return output, memory_loss

    def eval_with_adaptation(self, x, target=None, adaptation_steps=10):
        """
        Run evaluation with optional test-time adaptation on memory.
        """
        self.eval()
        # original_memory_state = copy.deepcopy(self.memory.state_dict())
        # before just to debug
        # with torch.no_grad():
        #     output, memory_loss, task_loss = self.forward(x, target=target, test = True)

        # Run adaptation steps on memory if desired
        for _ in range(adaptation_steps):
            # we do want grads here, so no torch.no_grad
            output, memory_loss = self.update_memory(x)
        # self.memory.load_state_dict(original_memory_state)
        # After adaptation, forward again to get updated output
        with torch.no_grad():
            output, memory_loss, task_loss = self.forward(x, target=target)
        # print(output, memory_loss, task_loss)
        return output, memory_loss, task_loss

############
# sanity check by implementing y = x + epislon

def generate_dummy_data(batch_size, seq_len, d_model, device='cpu'):
    """Generate synthetic data for regression task."""
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    # Simulate target as a noisy version of input
    target = x + torch.randn(batch_size, seq_len, d_model, device=device) * 0.1
    return x, target

def train(model, x, target, num_epochs=10):
    """Training loop with combined task and memory losses.
    Thus total loss = Task loss(the regression loss in this case) + memory loss(this would be trained via suprise) """
    model.train()
    for epoch in range(num_epochs):
        model.model_optimizer.zero_grad()
        output, memory_loss, task_loss = model.process_sequence(x, target=target) # we wont reset memory while training the Long memoery model
        total_loss = task_loss + memory_loss  # Combine losses
        total_loss.backward()
        model.model_optimizer.step()
        output_norm = torch.norm(output).item()
        print(f"Train Epoch {epoch}, Task Loss: {task_loss.item():.4f}, Memory Loss: {memory_loss.item():.4f}, Output Norm: {output_norm:.4f}")
    return model

def evaluate(model, x, target):
    """Evaluation loop with test-time adaptation (memory updates only)."""
    # print(model.eval_with_adaptation(x, target, adaptation_steps=1))
    output, memory_loss, task_loss = model.eval_with_adaptation(x, target, adaptation_steps=1)
    output_norm = torch.norm(output).item()
    print(f"Eval Task Loss: {task_loss.item():.4f}, Memory Loss: {memory_loss.item():.4f}, Output Norm: {output_norm:.4f}")
    # return task_loss.item(), memory_loss.item()
    return task_loss, memory_loss

def main():
    # Hyperparameters
    d_model = 512
    num_heads = 8
    num_persistent = 10
    segment_size = 64
    memory_hidden_dim = 256
    memory_num_layers = 2
    batch_size = 32
    seq_len = 1024
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = MAC(d_model, num_heads, num_persistent, segment_size, memory_hidden_dim, memory_num_layers)
    model.to(device)

    # Generate dummy data
    train_x, train_target = generate_dummy_data(batch_size, seq_len, d_model, device)
    test_x, test_target = generate_dummy_data(batch_size, seq_len, d_model, device)
    test_target +=2

    # Training phase
    print("Training Phase")
    model = train(model, train_x, train_target, num_epochs)

    # Evaluation phase
    print("\nEvaluation Phase")
    for i in range(1000):
        task_loss, memory_loss = evaluate(model, test_x, test_target)


if __name__ == "__main__":
    main()