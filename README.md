# minigrad
## A small autograd repo implementing backpropagation with ability to develop small Neural Networks.

### Greatly inspired by Andrej Karpathy's  [micrograd](https://github.com/karpathy/micrograd)

----------

#### TODO:

1) utils and visualizations

2) Cover all operations (ones highlighted are done):
```
`Relu`, Log, Exp                        # unary ops
Sum, Max                                # reduce ops (with axis argument)
`Add`, `Sub`, `Mul`, `Pow`              # binary ops (with broadcasting)
Reshape, Transpose, Slice               # movement ops
Matmul, Conv2D                          # processing ops
```




### Reverse Mode Auto DIfferentiation

Let's say we have expression 𝑧=𝑥1𝑥2+sin(𝑥1) and want to find derivatives 𝑑𝑧𝑑𝑥1 and 𝑑𝑧𝑑𝑥2. Reverse-mode AD splits this task into 2 parts, namely, forward and reverse passes.

#### Forward pass:

First step is to decompose the complex expression into a set of primitive ones, i.e. expressions consisting of at most single step or single function call.
$$
𝑤1=𝑥1\\

𝑤2=𝑥2\\

𝑤3=𝑤1𝑤2\\

𝑤4=sin(𝑤1)\\

𝑤5=𝑤3+𝑤4\\

𝑧=𝑤5\\
$$


The advantage of this representation is that differentiation rules for each separate expression are already known.

For example, we know that derivative of `sin` is `cos`, and so `dw4/dw1=cos⁡(w1)`.

We will use this fact in reverse pass below. Essentially, forward pass consists of evaluating each of these expressions and saving the results. 

Say, our inputs are: `𝑥1=2`  and `𝑥2=3`. Then we have:
$$
𝑤1=𝑥1=2\\

𝑤2=𝑥2=3\\

𝑤3=𝑤1𝑤2=6\\

𝑤4=sin(𝑤1) =0.9\\

𝑤5=𝑤3+𝑤4=6.9\\

𝑧=𝑤5=6.9
$$




#### Reverse pass:

This is the main part and it uses the **chain rule**.

In its basic form, chain rule states that if you have variable `𝑡(𝑢(𝑣))` which depends on `𝑢` which, in its turn, depends on `𝑣`, then:
$$
\\
𝑑𝑡/𝑑𝑣 = 𝑑𝑡/𝑑𝑢 * 𝑑𝑢/𝑑𝑣\\
$$
or, if `𝑡` depends on `𝑣` via several paths / variables `𝑢𝑖`, e.g.:
$$
𝑢1=𝑓(𝑣)\\

𝑢2=𝑔(𝑣)\\

𝑡=ℎ(𝑢1,𝑢2)\\
then:\\
𝑑𝑡/𝑑𝑣=\sum_{i} 𝑑𝑡/𝑑𝑢𝑖 *𝑑𝑢𝑖/𝑑𝑣
$$
In terms of expression graph, if we have a final node `𝑧` and input nodes `𝑤𝑖`, and path from `𝑧` to `𝑤𝑖` goes through intermediate nodes `𝑤𝑝` (i.e. 𝑧=𝑔(𝑤𝑝) where 𝑤𝑝=𝑓(𝑤𝑖)), we can find derivative `𝑑𝑧/𝑑𝑤𝑖` as


$$
𝑑𝑧/𝑑𝑤𝑖=\sum_{𝑝∈P𝑎𝑟𝑒𝑛𝑡𝑠(𝑖)}𝑑𝑧/𝑑𝑤𝑝 * 𝑑𝑤𝑝/𝑑𝑤𝑖
$$
In other words, to calculate the derivative of output variable 𝑧 w.r.t. any intermediate or input variable 𝑤𝑖, we only need to know the derivatives of its parents and the formula to calculate derivative of primitive expression 𝑤𝑝=𝑓(𝑤𝑖).



Reverse pass starts at the end (i.e. 𝑑𝑧/𝑑𝑧) and propagates backward to all dependencies. 
$$
𝑑𝑧/𝑑𝑧=1\\
$$
Then we know that 𝑧=𝑤5 and so:

​                                                                                                         𝑑𝑧/𝑑𝑤5 = 1

𝑤5 linearly depends on 𝑤3 and 𝑤4, so 𝑑𝑤5/𝑑𝑤3=1 and 𝑑𝑤5/𝑑𝑤4=1. Using the chain rule we find:
$$
𝑑𝑧/𝑑𝑤3=𝑑𝑧/𝑑𝑤5 × 𝑑𝑤5/ 𝑑𝑤3 = 1×1 = 1\\
 
𝑑𝑧/𝑑𝑤4=𝑑𝑧/𝑑𝑤5 × 𝑑𝑤5/𝑑𝑤4 = 1×1 = 1
$$


From definition 𝑤3=𝑤1𝑤2 and rules of partial derivatives, we find that 𝑑𝑤3 / 𝑑𝑤2=𝑤1. Thus:
$$
𝑑𝑧/𝑑𝑤2 = 𝑑𝑧/𝑑𝑤3 × 𝑑𝑤3/𝑑𝑤2 = 1 × 𝑤1 = 𝑤1
$$
Which, as we already know from forward pass, is:
$$
𝑑𝑧/𝑑𝑤2 = 𝑤1 = 2
$$


Finally, `𝑤1` contributes to `𝑧` via `𝑤3` and `𝑤4`. Once again, from the rules of partial derivatives we know that `𝑑𝑤3/𝑑𝑤1 = 𝑤2` and `𝑑𝑤4/𝑑𝑤1 = cos(𝑤1)`. Thus:
$$
𝑑𝑧/𝑑𝑤1 = 𝑑𝑧/𝑑𝑤3 * 𝑑𝑤3/𝑑𝑤1 + 𝑑𝑧/𝑑𝑤4 * 𝑑𝑤4/𝑑𝑤1 = 𝑤2 + cos(𝑤1)
$$


And again, given known inputs, we can calculate it:


$$
𝑑𝑧/𝑑𝑤1 = 𝑤2 + cos(𝑤1) = 3 + cos(2) = 2.58
$$


Since 𝑤1 and 𝑤2 are just aliases for 𝑥1 and 𝑥2, we get our answer:
$$
𝑑𝑧 / 𝑑𝑥1 = 2.58\\

 
𝑑𝑧 / 𝑑𝑥2 = 2
$$
