# reporters/terminal_reporter.py

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.progress import track
from rich import box
from rich.columns import Columns
from rich.text import Text

class TerminalReporter:
    """Prints beautiful terminal reports using Rich"""
    
    def __init__(self):
        self.console = Console()
    
    def print_report(self, result):
        """Print complete analysis report"""
        
        self.console.print()
        self.console.print(Panel.fit(
            "[bold cyan]🤖 AI ARCHITECTURE ANALYSIS REPORT[/bold cyan]",
            border_style="cyan"
        ))
        
        # Basic Info
        self._print_basic_info(result)
        
        # Parameter Count
        self._print_parameters(result)
        
        # Memory
        self._print_memory(result)
        
        # Layer Breakdown
        self._print_layers(result)
        
        # Performance
        if result.inference_time_ms > 0:
            self._print_performance(result)
        
        # Issues
        self._print_issues(result)
        
        # Recommendations
        self._print_recommendations(result)
    
    def _print_basic_info(self, result):
        
        info_table = Table(
            box=box.ROUNDED,
            show_header=False,
            border_style="blue"
        )
        
        info_table.add_column("Property", style="bold cyan")
        info_table.add_column("Value", style="white")
        
        info_table.add_row("Model Name", result.model_name or "Unknown")
        info_table.add_row("Framework", result.framework.upper())
        info_table.add_row("Model Type", result.model_type or "Unknown")
        info_table.add_row("Analysis Time", result.timestamp)
        info_table.add_row("Total Layers", str(result.layer_count))
        
        self.console.print(Panel(
            info_table,
            title="[bold]📋 BASIC INFORMATION[/bold]",
            border_style="blue"
        ))
    
    def _print_parameters(self, result):
        
        param_table = Table(box=box.ROUNDED, border_style="green")
        param_table.add_column("Type", style="bold")
        param_table.add_column("Count", style="green")
        param_table.add_column("Percentage", style="cyan")
        
        total = result.total_parameters
        
        param_table.add_row(
            "Total Parameters",
            f"{total:,}",
            "100%"
        )
        param_table.add_row(
            "Trainable",
            f"{result.trainable_parameters:,}",
            f"{result.trainable_parameters/total*100:.1f}%"
        )
        param_table.add_row(
            "Frozen",
            f"{result.frozen_parameters:,}",
            f"{result.frozen_parameters/total*100:.1f}%"
        )
        
        # Add size classification
        if total < 1e6:
            size_class = "Small Model (<1M params)"
        elif total < 1e9:
            size_class = "Medium Model (<1B params)"
        elif total < 1e11:
            size_class = "Large Model (<100B params)"
        else:
            size_class = "Very Large Model (100B+ params)"
        
        param_table.add_row(
            "Classification",
            size_class,
            ""
        )
        
        self.console.print(Panel(
            param_table,
            title="[bold]🔢 PARAMETER ANALYSIS[/bold]",
            border_style="green"
        ))
    
    def _print_memory(self, result):
        
        mem = result.memory_footprint
        
        mem_table = Table(box=box.ROUNDED, border_style="yellow")
        mem_table.add_column("Memory Type", style="bold")
        mem_table.add_column("Size", style="yellow")
        
        for key, value in mem.items():
            mem_table.add_row(
                key.replace('_', ' ').title(),
                f"{value:.2f} MB"
            )
        
        self.console.print(Panel(
            mem_table,
            title="[bold]💾 MEMORY FOOTPRINT[/bold]",
            border_style="yellow"
        ))
    
    def _print_layers(self, result):
        
        # Layer type summary
        type_table = Table(box=box.ROUNDED, border_style="magenta")
        type_table.add_column("Layer Type", style="bold magenta")
        type_table.add_column("Count", style="white")
        type_table.add_column("Visual", style="cyan")
        
        for layer_type, count in sorted(
            result.layer_breakdown.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if isinstance(count, int):
                bar = "█" * min(count, 20)
                type_table.add_row(layer_type, str(count), bar)
        
        self.console.print(Panel(
            type_table,
            title="[bold]🔬 LAYER BREAKDOWN[/bold]",
            border_style="magenta"
        ))
        
        # Detailed layer table (first 20)
        if result.layers:
            detail_table = Table(
                box=box.SIMPLE,
                border_style="white",
                show_header=True
            )
            detail_table.add_column("Layer Name", style="cyan", width=30)
            detail_table.add_column("Type", style="magenta", width=20)
            detail_table.add_column("Parameters", style="green", width=15)
            detail_table.add_column("Config", style="white", width=30)
            
            for layer in result.layers[:20]:
                detail_table.add_row(
                    layer['name'][:28],
                    layer['type'],
                    f"{layer['parameters']:,}",
                    str(layer.get('config', {}))[:28]
                )
            
            if len(result.layers) > 20:
                detail_table.add_row(
                    f"... and {len(result.layers)-20} more",
                    "", "", ""
                )
            
            self.console.print(Panel(
                detail_table,
                title="[bold]📊 LAYER DETAILS[/bold]",
                border_style="white"
            ))
    
    def _print_performance(self, result):
        
        perf_table = Table(box=box.ROUNDED, border_style="red")
        perf_table.add_column("Metric", style="bold red")
        perf_table.add_column("Value", style="white")
        perf_table.add_column("Rating", style="cyan")
        
        # Inference time rating
        ms = result.inference_time_ms
        if ms < 10:
            time_rating = "🟢 Excellent"
        elif ms < 50:
            time_rating = "🟡 Good"
        elif ms < 100:
            time_rating = "🟠 Acceptable"
        else:
            time_rating = "🔴 Slow - Optimize!"
        
        perf_table.add_row(
            "Inference Time",
            f"{ms:.2f} ms",
            time_rating
        )
        perf_table.add_row(
            "Throughput",
            f"{result.throughput:.1f} inferences/sec",
            ""
        )
        
        self.console.print(Panel(
            perf_table,
            title="[bold]⚡ PERFORMANCE METRICS[/bold]",
            border_style="red"
        ))
    
    def _print_issues(self, result):
        
        if not result.issues:
            self.console.print(Panel(
                "[green]✅ No issues found![/green]",
                title="[bold]🐛 ISSUES[/bold]",
                border_style="green"
            ))
            return
        
        issue_table = Table(box=box.ROUNDED, border_style="red")
        issue_table.add_column("Severity", style="bold", width=10)
        issue_table.add_column("Layer", style="cyan", width=20)
        issue_table.add_column("Issue", style="white", width=30)
        issue_table.add_column("Fix", style="green", width=30)
        
        for issue in result.issues:
            severity = issue['severity']
            color = "red" if severity == "ERROR" else \
                    "yellow" if severity == "WARNING" else "blue"
            
            issue_table.add_row(
                f"[{color}]{severity}[/{color}]",
                issue.get('layer', 'N/A'),
                issue.get('issue', ''),
                issue.get('fix', '')
            )
        
        self.console.print(Panel(
            issue_table,
            title=f"[bold]🐛 ISSUES FOUND: {len(result.issues)}[/bold]",
            border_style="yellow"
        ))
    
    def _print_recommendations(self, result):
        
        if not result.recommendations:
            return
        
        rec_text = "\n".join(
            f"  {i+1}. {rec}" 
            for i, rec in enumerate(result.recommendations)
        )
        
        self.console.print(Panel(
            rec_text,
            title="[bold]💡 RECOMMENDATIONS[/bold]",
            border_style="cyan"
        ))
