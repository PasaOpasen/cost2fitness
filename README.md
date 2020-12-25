# cost2fitness

PyPI package for conversion cost values (less is better) to fitness values (more is better) and vice versa

```
pip install cost2fitness
```

## About

This is the package containing several methods for transformation numpy arrays depended on scales, averages and so on. But the primary way to use it is the conversion from cost values (less is better) to fitness values (more is better) and vice versa. It can be highly helpful when u r using 

* evolutionary algorithms depended on numeric differences: so, it's important to set good representation of samples scores for better selection
* sampling methods with probabilities depended on samples scores 

## Transformers

There are several simple transformers. Each transformer is the subclass of `BaseTransformer` class containing `name` field and `transform(array)` method which transforms input array to new representation. 

Checklist:

* `ReverseByAverage`, 
* `AntiMax`, 
* `AntiMaxPercent(percent)`, 
* `Min2Zero`, 
* `Min2Value(value)`, 
* `ProbabilityView` (converts data to probabilities), 
* `SimplestReverse`, 
* `AlwaysOnes` (returns array of ones), 
* `NewAvgByMult(new_average)`,
* `NewAvgByShift(new_average)`

U can create your transformer using simple logic from [file](cost2fitness/transformers.py).

## Pipeline of transformers

## How to plot

## Examples

