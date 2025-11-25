
# AI-content-detector

A fast, local tool for detecting AI-generated English text using GPT-2 perplexity analysis. Ideal for academic submissions, originality checks, and avoiding conference rejections.

## Features
- Calculates average perplexity (PPL) and token loss variance for input text
- Provides a simple GUI for easy use
- Gives clear predictions: likely AI-generated, human-written, or mixed
- Runs locally—no usage limits or long waits
- Supports GPT-2, GPT-2 Medium, and GPT-2 Large

## Installation

1. Clone the repository:
	```bash
	git clone https://github.com/<your-username>/AI-content-detector.git
	cd AI-content-detector
	```
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

## Usage

Run the detector with:
```bash
python AI_detector.py
```

- Enter your English text in the GUI and click "計算困惑度" (Calculate Perplexity).
- The tool will display average perplexity, token loss variance, and a prediction about the text's origin.

## How It Works
- Uses GPT-2 to compute perplexity and token-level loss variance.
- Lower perplexity and variance often indicate AI-generated text; higher values suggest human writing.
- Thresholds are empirical and may be adjusted for different use cases.

## Example
![Screenshot](readme_files/image.png)

## Contributing
We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Use Conventional Commits for commit messages.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

> [!IMPORTANT] 
> *Disclaimer: Results are for reference only. The tool does not guarantee detection accuracy.*