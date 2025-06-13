# Language Model Benchmark Framework

A comprehensive framework for evaluating language models on open-ended questions, inspired by AidanBench. This framework measures model performance across multiple dimensions including creativity, coherence, and novelty.

## Features

- Multi-dimensional evaluation (coherence, novelty, creativity)
- Configurable evaluation parameters
- Comprehensive visualization of results
- Support for multiple models
- Detailed question-level analysis
- Automated scoring using judge models
- Embedding-based novelty detection

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

1. Prepare your questions file (one question per line):
```text
What architectural features might you include in a tasteful house?
How could we redesign schools to better prepare students for the 22nd century?
What activities might I include at a party for firefighters?
```

2. Run the benchmark:
```bash
python benchmark/main.py \
    --questions path/to/questions.txt \
    --models gpt-4 gpt-3.5-turbo \
    --temperature 0.7 \
    --coherence-threshold 15.0 \
    --novelty-threshold 0.15
```

## Output

The framework generates:
- JSON results file with detailed metrics
- Comparison plots across models
- Question-level breakdowns for each model
- Coherence and novelty score distributions

## Project Structure

```
benchmark/
├── core/
│   └── evaluator.py      # Core evaluation logic
├── metrics/
│   └── scoring.py        # Scoring and metrics
├── visualization/
│   └── plotter.py        # Results visualization
├── main.py              # Main entry point
└── utils/               # Utility functions

data/
├── raw/                 # Raw question files
└── processed/           # Processed data

results/
├── raw/                 # Raw evaluation results
└── processed/           # Processed results and plots

tests/
├── unit/               # Unit tests
└── integration/        # Integration tests
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{lm-benchmark-framework,
  author = {Your Name},
  title = {Language Model Benchmark Framework},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/lm-benchmark-framework}
}
``` 