# main.py

from core.analyzer import AIArchitectureAnalyzer

def main():
    """
    Example usage showing how to analyze different model types
    """
    
    print("🤖 AI Architecture Analyzer")
    print("=" * 60)
    
    analyzer = AIArchitectureAnalyzer(verbose=True)
    
    # ─────────────────────────────────────────
    # EXAMPLE 1: Analyze a HuggingFace LLM
    # ─────────────────────────────────────────
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    result = analyzer.analyze(
        model,
        input_shape=(1, 512),  # batch_size, sequence_length
    )
    analyzer.report(result)
    
    # ─────────────────────────────────────────
    # EXAMPLE 2: Analyze your own PyTorch model
    # ─────────────────────────────────────────
    import torch
    import torch.nn as nn
    
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(50000, 512)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(512, 8),
                num_layers=6
            )
            self.fc = nn.Linear(512, 50000)
        
        def forward(self, x):
            x = self.embed(x)
            x = self.transformer(x)
            return self.fc(x)
    
    my_model = MyModel()
    sample = torch.randint(0, 50000, (1, 128))
    
    result = analyzer.analyze(
        my_model,
        sample_input=sample
    )
    analyzer.report(result)

if __name__ == "__main__":
    main()
