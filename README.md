# FP-TBF-DI

Reference implementation for **基于局部一致规则的乳腺超声病灶逐批分类方法**
(Sequential Classification of Breast Ultrasound Lesions Using Locally Consistent Rules).

With Python 3.12, run these commands from the repository root:

```sh
cd fp-tbf-di-code
python -m pip install -r requirements.txt
python demo.py
```

The demo generates its inputs locally. See the [implementation guide](fp-tbf-di-code/README.md)
for the command-line runner and input format. The clinical score table, study
split indices, and per-record study predictions are outside the public repository.
