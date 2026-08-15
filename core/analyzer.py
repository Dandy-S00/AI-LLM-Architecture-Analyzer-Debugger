# core/analyzer.py

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import time

@dataclass
class AnalysisResult:
    """Stores complete analysis results"""
    
    # Basic Info
    model_name: str = ""
    model_type: str = ""
    framework: str = ""
    timestamp: str = ""
    
    # Architecture
    total_parameters: int = 0
    trainable_parameters: int = 0
    frozen_parameters: int = 0
    layer_count: int = 0
    layer_breakdown: Dict = field(default_factory=dict)
    
    # Memory
    model_size_mb: float = 0.0
    memory_footprint: Dict = field(default_factory=dict)
    
    # Performance
    inference_time_ms: float = 0.0
    throughput: float = 0.0
    flops: int = 0
    
    # Issues Found
    issues: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Full Layer Details
    layers: List[Dict] = field(default_factory=list)


class AIArchitectureAnalyzer:
    """
    Main analyzer class - handles any AI/LLM architecture
    
    Usage:
        analyzer = AIArchitectureAnalyzer()
        result = analyzer.analyze(your_model)
        analyzer.report(result)
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.result = AnalysisResult()
        self.result.timestamp = datetime.now().isoformat()
        
        # Import parsers
        from parsers.pytorch_parser import PyTorchParser
        from parsers.huggingface_parser import HuggingFaceParser
        from parsers.onnx_parser import ONNXParser
        
        self.parsers = {
            'pytorch': PyTorchParser(),
            'huggingface': HuggingFaceParser(),
            'onnx': ONNXParser()
        }
    
    def analyze(
        self, 
        model: Any,
        input_shape: Optional[tuple] = None,
        sample_input: Optional[Any] = None,
        deep_analysis: bool = True
    ) -> AnalysisResult:
        """
        Main entry point - analyzes any AI model
        
        Args:
            model: Your AI model (any framework)
            input_shape: Optional input shape for profiling
            sample_input: Optional sample input for testing
            deep_analysis: Run full deep analysis
        
        Returns:
            AnalysisResult with complete findings
        """
        
        print("🔍 Starting AI Architecture Analysis...")
        print("=" * 60)
        
        # Step 1: Detect framework
        framework = self._detect_framework(model)
        self.result.framework = framework
        print(f"✅ Framework detected: {framework}")
        
        # Step 2: Parse architecture
        print("📊 Parsing architecture...")
        self._parse_architecture(model, framework)
        
        # Step 3: Count parameters
        print("🔢 Counting parameters...")
        self._count_parameters(model, framework)
        
        # Step 4: Analyze layers
        print("🔬 Analyzing layers...")
        self._analyze_layers(model, framework)
        
        # Step 5: Memory analysis
        print("💾 Analyzing memory footprint...")
        self._analyze_memory(model, framework)
        
        # Step 6: Performance profiling
        if sample_input or input_shape:
            print("⚡ Profiling performance...")
            self._profile_performance(
                model, 
                framework,
                input_shape, 
                sample_input
            )
        
        # Step 7: Deep analysis
        if deep_analysis:
            print("🧠 Running deep analysis...")
            self._deep_analysis(model, framework)
        
        # Step 8: Find issues
        print("🐛 Checking for issues...")
        self._find_issues(model, framework)
        
        # Step 9: Generate recommendations
        print("💡 Generating recommendations...")
        self._generate_recommendations()
        
        print("=" * 60)
        print("✅ Analysis complete!")
        
        return self.result
    
    def _detect_framework(self, model: Any) -> str:
        """Auto-detect which AI framework the model uses"""
        
        # Check PyTorch
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                return "pytorch"
        except ImportError:
            pass
        
        # Check TensorFlow/Keras
        try:
            import tensorflow as tf
            if isinstance(model, tf.keras.Model):
                return "tensorflow"
            if isinstance(model, tf.Module):
                return "tensorflow"
        except ImportError:
            pass
        
        # Check HuggingFace
        try:
            from transformers import PreTrainedModel
            if isinstance(model, PreTrainedModel):
                return "huggingface"
        except ImportError:
            pass
        
        # Check ONNX
        try:
            import onnx
            if isinstance(model, onnx.ModelProto):
                return "onnx"
        except ImportError:
            pass
        
        # Check if it's a file path
        if isinstance(model, str):
            return self._detect_from_path(model)
        
        return "unknown"
    
    def _detect_from_path(self, path: str) -> str:
        """Detect framework from file extension"""
        
        extensions = {
            '.pt': 'pytorch',
            '.pth': 'pytorch', 
            '.h5': 'tensorflow',
            '.pb': 'tensorflow',
            '.onnx': 'onnx',
            '.json': 'huggingface',
            '.bin': 'huggingface'
        }
        
        for ext, framework in extensions.items():
            if path.endswith(ext):
                return framework
        
        return "unknown"
    
    def _count_parameters(self, model: Any, framework: str):
        """Count total, trainable and frozen parameters"""
        
        if framework == "pytorch":
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(
                p.numel() for p in model.parameters() 
                if p.requires_grad
            )
            
            self.result.total_parameters = total
            self.result.trainable_parameters = trainable
            self.result.frozen_parameters = total - trainable
            
        elif framework in ["tensorflow", "keras"]:
            self.result.total_parameters = model.count_params()
            trainable = sum(
                tf.size(w).numpy() 
                for w in model.trainable_weights
            )
            self.result.trainable_parameters = trainable
            self.result.frozen_parameters = (
                self.result.total_parameters - trainable
            )
    
    def _analyze_layers(self, model: Any, framework: str):
        """Deep dive into every layer"""
        
        layers = []
        layer_types = {}
        
        if framework == "pytorch":
            for name, module in model.named_modules():
                
                # Skip container modules
                if len(list(module.children())) > 0:
                    continue
                
                layer_type = type(module).__name__
                params = sum(p.numel() for p in module.parameters())
                
                layer_info = {
                    'name': name,
                    'type': layer_type,
                    'parameters': params,
                    'input_shape': None,  # filled during profiling
                    'output_shape': None,
                    'activation': self._detect_activation(module),
                    'has_bias': hasattr(module, 'bias') and module.bias is not None,
                    'config': self._get_layer_config(module)
                }
                
                layers.append(layer_info)
                
                # Count layer types
                layer_types[layer_type] = layer_types.get(
                    layer_type, 0
                ) + 1
        
        self.result.layers = layers
        self.result.layer_count = len(layers)
        self.result.layer_breakdown = layer_types
    
    def _detect_activation(self, module: Any) -> str:
        """Detect activation function used in a layer"""
        
        import torch.nn as nn
        
        activation_map = {
            nn.ReLU: "ReLU",
            nn.GELU: "GELU",
            nn.SiLU: "SiLU/Swish",
            nn.Tanh: "Tanh",
            nn.Sigmoid: "Sigmoid",
            nn.Softmax: "Softmax",
            nn.LeakyReLU: "LeakyReLU",
            nn.ELU: "ELU"
        }
        
        for activation_class, name in activation_map.items():
            if isinstance(module, activation_class):
                return name
        
        return "None"
    
    def _get_layer_config(self, module: Any) -> Dict:
        """Extract configuration from a layer"""
        
        import torch.nn as nn
        config = {}
        
        # Linear layers
        if isinstance(module, nn.Linear):
            config = {
                'in_features': module.in_features,
                'out_features': module.out_features
            }
        
        # Convolutional layers
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            config = {
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding
            }
        
        # Attention layers (Transformers)
        elif isinstance(module, nn.MultiheadAttention):
            config = {
                'embed_dim': module.embed_dim,
                'num_heads': module.num_heads,
                'dropout': module.dropout
            }
        
        # Normalization
        elif isinstance(module, nn.LayerNorm):
            config = {
                'normalized_shape': module.normalized_shape
            }
        
        elif isinstance(module, nn.BatchNorm2d):
            config = {
                'num_features': module.num_features,
                'eps': module.eps,
                'momentum': module.momentum
            }
        
        # Embedding layers
        elif isinstance(module, nn.Embedding):
            config = {
                'num_embeddings': module.num_embeddings,
                'embedding_dim': module.embedding_dim
            }
        
        return config
    
    def _analyze_memory(self, model: Any, framework: str):
        """Calculate memory usage"""
        
        import sys
        
        if framework == "pytorch":
            import torch
            
            # Calculate parameter memory
            param_memory = sum(
                p.nelement() * p.element_size() 
                for p in model.parameters()
            )
            
            buffer_memory = sum(
                b.nelement() * b.element_size() 
                for b in model.buffers()
            )
            
            total_mb = (param_memory + buffer_memory) / 1024 / 1024
            
            self.result.model_size_mb = total_mb
            self.result.memory_footprint = {
                'parameters_mb': param_memory / 1024 / 1024,
                'buffers_mb': buffer_memory / 1024 / 1024,
                'total_mb': total_mb,
                'estimated_inference_mb': total_mb * 2,
                'estimated_training_mb': total_mb * 4
            }
    
    def _profile_performance(
        self, 
        model: Any, 
        framework: str,
        input_shape: Optional[tuple],
        sample_input: Optional[Any]
    ):
        """Profile inference speed and throughput"""
        
        if framework == "pytorch":
            import torch
            
            model.eval()
            
            # Create dummy input if needed
            if sample_input is None and input_shape:
                sample_input = torch.randn(*input_shape)
            
            # Warmup runs
            print("  Running warmup passes...")
            with torch.no_grad():
                for _ in range(3):
                    _ = model(sample_input)
            
            # Timing runs
            times = []
            num_runs = 10
            
            print(f"  Running {num_runs} timed passes...")
            with torch.no_grad():
                for _ in range(num_runs):
                    start = time.perf_counter()
                    output = model(sample_input)
                    end = time.perf_counter()
                    times.append((end - start) * 1000)
            
            avg_time = np.mean(times)
            self.result.inference_time_ms = avg_time
            self.result.throughput = 1000 / avg_time  # inferences/sec
    
    def _deep_analysis(self, model: Any, framework: str):
        """Run deep analysis specific to model type"""
        
        # Detect if it's a transformer/LLM
        if self._is_transformer(model, framework):
            self._analyze_transformer(model, framework)
        
        # Detect if it's a CNN
        elif self._is_cnn(model, framework):
            self._analyze_cnn(model, framework)
    
    def _is_transformer(self, model: Any, framework: str) -> bool:
        """Check if model is a transformer architecture"""
        
        if framework == "pytorch":
            import torch.nn as nn
            # Check for attention layers
            for module in model.modules():
                if isinstance(module, nn.MultiheadAttention):
                    return True
                if 'attention' in type(module).__name__.lower():
                    return True
        return False
    
    def _is_cnn(self, model: Any, framework: str) -> bool:
        """Check if model is a CNN architecture"""
        
        if framework == "pytorch":
            import torch.nn as nn
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    return True
        return False
    
    def _analyze_transformer(self, model: Any, framework: str):
        """Special analysis for transformer models"""
        
        if framework == "pytorch":
            import torch.nn as nn
            
            attention_layers = []
            
            for name, module in model.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    attention_layers.append({
                        'name': name,
                        'heads': module.num_heads,
                        'embed_dim': module.embed_dim,
                        'head_dim': module.embed_dim // module.num_heads
                    })
            
            if attention_layers:
                self.result.layer_breakdown['attention_analysis'] = {
                    'total_attention_layers': len(attention_layers),
                    'layers': attention_layers
                }
    
    def _find_issues(self, model: Any, framework: str):
        """Automatically detect common issues"""
        
        issues = []
        
        if framework == "pytorch":
            import torch.nn as nn
            
            # Check for potential vanishing gradient
            for name, module in model.named_modules():
                if isinstance(module, nn.Sigmoid):
                    issues.append({
                        'severity': 'WARNING',
                        'layer': name,
                        'issue': 'Sigmoid activation can cause vanishing gradients',
                        'fix': 'Consider using ReLU or GELU instead'
                    })
                
                # Check for missing batch norm after conv
                if isinstance(module, nn.Conv2d):
                    issues.append({
                        'severity': 'INFO',
                        'layer': name,
                        'issue': 'Check if BatchNorm follows Conv layer',
                        'fix': 'Add BatchNorm2d after Conv2d for stable training'
                    })
                
                # Check for large linear layers
                if isinstance(module, nn.Linear):
                    if module.in_features > 10000:
                        issues.append({
                            'severity': 'WARNING',
                            'layer': name,
                            'issue': f'Very large linear layer: {module.in_features}',
                            'fix': 'Consider dimensionality reduction'
                        })
            
            # Check parameter count
            if self.result.total_parameters > 1_000_000_000:
                issues.append({
                    'severity': 'INFO',
                    'layer': 'global',
                    'issue': f'Very large model: {self.result.total_parameters:,} params',
                    'fix': 'Consider quantization or pruning for deployment'
                })
        
        self.result.issues = issues
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        
        recs = []
        
        # Memory recommendations
        if self.result.model_size_mb > 1000:
            recs.append(
                "💾 Large model detected. Consider INT8 quantization "
                "to reduce size by ~4x with minimal accuracy loss"
            )
        
        # Parameter recommendations  
        if self.result.frozen_parameters > self.result.trainable_parameters:
            recs.append(
                "🔧 Most parameters are frozen. This looks like "
                "fine-tuning. Consider LoRA for more efficient training"
            )
        
        # Performance recommendations
        if self.result.inference_time_ms > 100:
            recs.append(
                "⚡ Inference > 100ms. Consider: TorchScript, "
                "ONNX export, or TensorRT for production deployment"
            )
        
        # Issue-based recommendations
        for issue in self.result.issues:
            if issue['severity'] == 'WARNING':
                recs.append(f"⚠️ {issue['layer']}: {issue['fix']}")
        
        self.result.recommendations = recs
    
    def report(self, result: AnalysisResult = None):
        """Print beautiful formatted report"""
        
        if result is None:
            result = self.result
        
        from reporters.terminal_reporter import TerminalReporter
        reporter = TerminalReporter()
        reporter.print_report(result)
