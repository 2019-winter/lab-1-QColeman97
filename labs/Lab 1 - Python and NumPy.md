---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
Quinn Coleman


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


**YOUR ANSWER HERE** - later


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.


## Exercise 1

```python
import numpy as np

a = np.full((6, 4), 2)
a
```

## Exercise 2

```python
b = np.ones((6, 4), dtype=int)
np.fill_diagonal(b, 3)
b
```

## Exercise 3

```python
a * b
# np.dot(a,b)
```

- dot(a,b) doesn't work because that is a dot product that requires matrices a and b to have equal inner dimensions.
- a * b works because that is an element-wise multiplication and matrices a and b have the same dimensions.


## Exercise 4

```python
a
a.transpose()
b
b.transpose()
display(np.dot(a.transpose(), b))
np.dot(a, b.transpose())
```

The shapes are different because they are results of different dot products.


## Exercise 5

```python
def cool_func():
    print('This is a pretty cool function')
    
cool_func()
```

## Exercise 6

```python
def do_matrix_stuff():
    a = np.random.rand(3,4)
    b = np.random.rand(2,5)
    print('Sum of a:', np.sum(a)) 
    print('Sum of b:', b.sum()) # Same op - Python built-in
    print('Mean of a:', np.mean(b))
    print('Mean of b:', b.mean()) # Same op - Python built-in
    print('Median of a:', np.median(a))
    print('Median of b:', np.median(b)) # No Python built-in
    
do_matrix_stuff()
```

## Exercise 7

```python
def count_ones(array):
    counter = 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] == 1:
                counter += 1
    return counter

         
a = np.arange(9).reshape(3,3)

display(count_ones(a))
np.sum(np.where(a == 1, 1, 0))
```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.

```python
import pandas as pd

a_matrix = np.full((6, 4), 2)
a = pd.DataFrame(a_matrix)
a
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.

```python
b_matrix = np.ones((6, 4), dtype=int)
b_matrix[np.arange(4), np.arange(4)] = 3
b = pd.DataFrame(b_matrix)
b
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.

```python
a * b
# a.dot(b)
```

The same results happen, for the same reasons as with the numpy array implementation:
- dot(a,b) doesn't work because that is a dot product that requires matrices a and b to have equal inner dimensions.
- a * b works because that is an element-wise multiplication and matrices a and b have the same dimensions.


## Exercise 11
Repeat exercise A.7 using a dataframe.

```python
# Can use np.where or pd.where
def count_ones(dataframe):
    counter = 0
    for i in range(dataframe.shape[0]):
        for j in range(dataframe.shape[1]):
            if dataframe.iloc[i,j] == 1:
                counter += 1
    return counter

a = pd.DataFrame(np.arange(9).reshape(3,3))

display(count_ones(a))
a.where(a == 1, 0).sum().sum()
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df.head()
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
titanic_df.name
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
titanic_df.set_index('sex',inplace=True)
len(titanic_df.loc['female'])
```

## Exercise 14
How do you reset the index?

```python
titanic_df.reset_index(inplace=True)
titanic_df.head()
```
