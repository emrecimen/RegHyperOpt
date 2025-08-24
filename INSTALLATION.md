# Installation Guide

## 1. Clone the Repository
```bash
git clone https://github.com/yourusername/regHyperOpt-demo.git
cd regHyperOpt-demo
```

## 2. (Optional) Create a Virtual Environment
It is recommended to use a virtual environment to keep dependencies isolated.  
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

## 3. Install Dependencies
Install the required Python packages:  
```bash
pip install tensorflow keras scikit-learn numpy matplotlib pandas
```

*(If you want exact versions, create a `requirements.txt` with pinned versions and run `pip install -r requirements.txt`.)*

## 4. Run the Example
Simply execute the demo script:  
```bash
python example.py
```

This will:  
- Load the MNIST dataset  
- Run the optimizer with a small search space  
- Print the best hyperparameter combination found  

## 5. Import into Your Own Project
You can import the optimizer into your own Python scripts:  
```python
from regHyperOpt import ml_optimizer, create_model
```
