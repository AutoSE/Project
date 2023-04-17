# Cluster

## Clustering

"Clustering" means grouping together  similar things
- Give goal attributes $Y$  and other attributes $X$ it is typically groupings in $X$ space.
  - Distance calculations in $Y$ space is usually called "domination" (see below)
- But as we shall see, clustering in $X$, pls a little sampling in $Y$ can be very useful.


## Relevance to SE


Applications:
- as a runtime checker (cluster the data, then report any new inputs that fall outside the range of the clusters of the test data used to test the system)[^liu]
- as an optimization tool (don't explore the whole space; instead cluster and run your analysis on many small clusters)[^maj18][^riot]
- as an optimizer in its own right (see below)[^chen]
- as a test generation tool (to partition the input space, then run just a few tests per partition) [^stall]
- as a verification tool to visualize the output space of a system[^davies] 
- as a requirements engineering tool (to encourage a fast discussion across the whole space)[^leit]
  - In the summer of 2011 and 2012,  I spent two months working on-site at Microsoft Redmond, observing data mining analysts.
    - I observed numerous meetings where Microsoft’s data scientists and business users discussed logs of defect data. 
    - There was a surprising little inspection of the output of data miners as compared to another process, 
        which we might call _fishing_. 
    - In fishing, analysts and users spend much time inspecting and discussing small samples of either raw or exemplary or synthesized project data. 
    - For example, in _data engagement meetings_, users debated the implications of data displayed on a screen. 
    -  In this way, users engaged with the data and with each other by monitoring each other’s queries and checking each other’s conclusions.
- As a privacy tool (why share all the data? why not just cluster and just a few cluster centroids?)[^peters]
  - [Fayola Peters](https://www.ezzoterik.com/papers/15lace2.pdf) used cluster + contrast to prune, as she passed data around a community. 
   - For example, in the following, green rows are those nearest the cluster centroids and blue rows are the ones most associated with the last column
       (bugs/10Kloc).
   - Discard things are aren't blue of green. 
   - She ended up sharing 20% of the rows and around a third of the columns. 1 - 1/5\*1/3 thus offered 93%   privacy
   - As for the remaining 7% of the data, we ran a mutator that pushed up items up the boundary point between classes (and no further). Bu certain common measures of privacy, that made the 7% space 80% private. 
   - Net effect 93% + .8*7 = 98.4% private,
   - And, FYI, inference on the tiny green+blue region was as effective as inference over all

<!-- <img width=700 src="/etc/img/peters1.png"> -->


<!-- <img width=700 src="/etc/img/peters2.png"> -->


[^peters]: Peters, Fayola, Tim Menzies, and Lucas Layman.](https://www.ezzoterik.com/papers/15lace2.pdf)
    2015 IEEE/ACM 37th IEEE International Conference on Software Engineering. Vol. 1. IEEE, 2015.


[^davies]: Davies, Misty, and Karen Gundy-Burlet. 
  ["Visualization of Global Sensitivity Analysis Results Based on a Combination of Linearly Dependent and Independent Directions."](https://ntrs.nasa.gov/api/citations/20110010856/downloads/20110010856.pdf)
  AIAA Infotech@ Aerospace 2010. 2010. 3387.


[^stall]: Dimitri Stallenberg, Mitchell Olsthoorn, and Annibale Panichella. 2022. 
 [Improving test case generation for REST APIs through hierarchical clustering] https://chinagator.github.io/papers/J5.pdf)
 In Proceedings of the 36th IEEE/ACM International Conference on Automated Software Engineering (ASE '21). IEEE Press, 117–128. https://doi.org/10.1109/ASE51524.2021.9678586


[^maj18]: Suvodeep Majumder, Nikhila Balaji, Katie Brey, Wei Fu, and Tim Menzies. 2018. 
[500+ times faster than deep learning: a case study exploring faster methods for text mining stackoverflow](https://arxiv.org/pdf/1802.05319.pdf). 
In Proceedings of the 15th International Conference on Mining Software Repositories (MSR '18). Association for Computing Machinery, New York, NY, USA, 554–563. https://doi.org/10.1145/3196398.3196424


[^leit]: Veerappa, Varsha, and Emmanuel Letier. 
  ["Understanding clusters of optimal solutions in multi-objective decision problems."](http://www0.cs.ucl.ac.uk/staff/e.letier/publications/2011-clusteringSolutions.pdf)
  2011 IEEE 19Th international requirements engineering conference. IEEE, 2011.


[^liu]: Liu, Z., Qin, T., Guan, X., Jiang, H., & Wang, C. (2018). 
  [An integrated method for anomaly detection from massive system logs](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8371223)
  IEEE Access, 6, 30602-30611.


[^riot]:  Jianfeng Chen, Tim Menzies:
  [RIOT: A Stochastic-Based Method for Workflow Scheduling in the Cloud](https://arxiv.org/pdf/1708.08127.pdf)
  IEEE CLOUD 2018: 318-325

[^chen]: J. Chen, V. Nair, R. Krishna and T. Menzies, 
 ["Sampling” as a Baseline Optimizer for Search-Based Software Engineering,"](https://arxiv.org/pdf/1608.07617.pdf)
 in IEEE Transactions on Software Engineering, vol. 45, no. 6, pp. 597-614, 1 June 2019, doi: 10.1109/TSE.2018.2790925.


## The Core of Clustering


Large amounts of data can be approximated by the centroids of a few clusters. Why?
- "In most applications examples are not spread uniformly throughout the instance space, but are concentrated on or near
  a lower-dimensional manifold. Learners can implicitly take
  advantage of this lower effective dimension."      
  -- Pedro Domingoes
- Also,  Many rows must be similar to the point of redundancy  since
 - when we build a model, 
  each part of that model should have support from multiple
  data points. 
 - This means that all the rows can be shrunk back to just a few examples.


For example, here we are clustering 398 examples of cars using the $X$ variables: 
- find two distant examples $A,B$
- divide other examples into those closest to $A$ or $B$
- recurse on each half
- As a result, we can approximate 398 examples with just 16.


This data has three $Y$ variables, acceleration (which we want to maximize), weight (which
we want to minimize) and miles per hour (which we want to maximize). We print the mean of these
values at the root and at each leaf:
- Note that even though we are not trying to, our clusters do separate good from bad $Y$ values:

```
398  {:Acc+ 15.6 :Lbs- 2970.4 :Mpg+ 23.8}
| 199
| | 99
| | | 49
| | | | 24  {:Acc+ 17.3 :Lbs- 2623.5 :Mpg+ 30.4}
| | | | 25  {:Acc+ 16.3 :Lbs- 2693.4 :Mpg+ 29.2}
| | | 50
| | | | 25  {:Acc+ 15.8 :Lbs- 2446.1 :Mpg+ 27.2}
| | | | 25  {:Acc+ 16.7 :Lbs- 2309.2 :Mpg+ 26.0}
| | 100
| | | 50
| | | | 25  {:Acc+ 16.2 :Lbs- 2362.5 :Mpg+ 32.0}
| | | | 25  {:Acc+ 16.4 :Lbs- 2184.1 :Mpg+ 34.8}
| | | 50
| | | | 25  {:Acc+ 16.2 :Lbs- 2185.8 :Mpg+ 29.6} <== best?
| | | | 25  {:Acc+ 16.3 :Lbs- 2179.4 :Mpg+ 26.4}
| 199
| | 99
| | | 49
| | | | 24  {:Acc+ 16.6 :Lbs- 2716.9 :Mpg+ 22.5}
| | | | 25  {:Acc+ 16.1 :Lbs- 3063.5 :Mpg+ 20.4}
| | | 50
| | | | 25  {:Acc+ 17.4 :Lbs- 3104.6 :Mpg+ 21.6}
| | | | 25  {:Acc+ 16.3 :Lbs- 3145.6 :Mpg+ 22.0}
| | 100
| | | 50
| | | | 25  {:Acc+ 12.4 :Lbs- 4320.5 :Mpg+ 12.4}
| | | | 25  {:Acc+ 11.3 :Lbs- 4194.2 :Mpg+ 12.8} <== worst
| | | 50
| | | | 25  {:Acc+ 13.7 :Lbs- 4143.1 :Mpg+ 18.0}
| | | | 25  {:Acc+ 14.4 :Lbs- 3830.2 :Mpg+ 16.4}
```
Now here's nearly the same algorithm, but know we run a   greedy search over the splits.
When splitting on  two distance points $A,B$, we peel at the $Y$ values  
 and ignore the worse half. 
```
398  {:Acc+ 15.6 :Lbs- 2970.4 :Mpg+ 23.8}
| 199
| | 100
| | | 50
| | | | 25  {:Acc+ 17.2 :Lbs- 2001.0 :Mpg+ 33.2}
```
Note that:
- This "clustering" algorithm is now an optimizer since it can isolate the best 25 examples out of the 398
- And it does so after evaluating just 5 examples
 - two at the root, 
 - then for each sub-split, we reuse one of the $A,B$ from the parent (the one that was best)


(Aside: the leaf node found via our optimizer never appears in the cluster tree. Why?)

# How to Cluster


## Distance (Basic)
 Here is Aha's instance-based distance algorithm,
[section 2.4](https://link.springer.com/content/pdf/10.1007/BF00153759.pdf).


(Note: slow. Ungood for large dimensional spaces. We'll fix that below.)


L-norm (L2):


- D= (&sum; (&Delta;(x,y))<sup>p</sup>))<sup>1/p</sup>
- euclidean : p=2


But what is &Delta;  ?


-  &Delta; Symbols: 
    - return 0 if x == y else 1
- &Delta;  Numbers:
    -  x - y
    - to make numbers fair with symbols, normalize x,y 0,1 using (x-min)/(max-min)


What about missing values:


- assume worst case
- if both unknown, assume &delta; = 1
- if one symbol missing, assume &delta; = 1
- if one number missing:
    - let x,y be unknown, known
    - y = normalize(y)
    - x = 0 if y > 0.5 else 1
    - &Delta; =  (x-y)


In the following recall that `DaATA` keeps column headers separately
for the `i.cols.x` (independent) columns and the
`i.cols.y` (dependent) columns. 


```lua
function DATA.dist(i,row1,row2,  cols,      n,d) --> n; returns 0..1 distance `row1` to `row2`
  n,d = 0,0
  for _,col in pairs(cols or i.cols.x) do
    n = n + 1
    d = d + col:dist(row1.cells[col.at], row2.cells[col.at])^the.p end
  return (d/n)^(1/the.p) end


function SYM.dist(i,s1,s2)
  return s1=="?" and s2=="?" and 1 or (s1==s2) and 0 or 1 end 


function NUM.dist(i,n1,n2)
  if n1=="?" and n2=="?" then return 1 end -- here's the AHA assumption (assume the max)
  n1,n2 = i:norm(n1), i:norm(n2)
  if n1=="?" then n1 = n2<.5 and 1 or 0 end -- here's the AHA assumption (assume the max)
  if n2=="?" then n2 = n1<.5 and 1 or 0 end -- here's the AHA assumption (assume the max)
  return math.abs(n1 - n2) end 


function NUM.norm(i,n)
  return n == "?" and n  or (n - i.lo)/(i.hi - i.lo + 1E-32) end
```
With the above, we can sort other rows by their distance to `row1`:
```lua
function DATA.around(i,row1,  rows,cols) --> t; sort other `rows` by distance to `row`
  return sort(map(rows or i.rows, 
                  function(row2)  return {row=row2, dist=i:dist(row1,row2,cols)} end),lt"dist") end
```
In the above, `sort` is a table sort function that controls the sorting via a second sorting function.
In this case, the secondary function is `lt` which is a function that returns a function that sorts
items ascending on some argument:


```lua
function lt(x) --> fun;  return a function that sorts ascending on `x`
  return function(a,b) return a[x] < b[x] end end


function sort(t, fun) --> t; return `t`,  sorted by `fun` (default= `<`)
  table.sort(t,fun); return t end
```
### K-means


<img src="https://dashee87.github.io/images/kmeans.gif" width=600 align=right>


k-means:
- pick $k$ random points
- labeling everyone by their closest $k$
- move $k$ to mean value of all points with same label


E.g. [mini-batch k-means](https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)
- Pick first 20 examples (at random) as _controids_
- for everything else, run through in batches of size (say 500,2000,etc)
  - mark each item with its nearest centroid
  - between batch B and B+1, move centroid towards its marked examples
    - "n" stores how often this centroid has been picked by new data
    - Each item "pulls" its centroid  attribute "c" towards its own attribute "x"  by an amount weighted   $c = (1-1/n)*c + x/n$. 
    - Note that when "n" is large, "c" barely moves at all.
   
## Recursive Bi-Clustering


e.g. recursive k-means with k=2.


But there are many more:


### KD-Treess
- Look at all the attributes
- Findthe one with the greatest diversity
- Split on its central point (e.g. mean, median, mode)
- Recurse on both halves.

```python
from collections import namedtuple
from operator import itemgetter
from pprint import pformat


class Node(namedtuple("Node", "location left_child right_child")):
    def __repr__(self):
        return pformat(tuple(self))


def kdtree(point_list, depth: int = 0):
    if not point_list:
        return None


    k = len(point_list[0])  # assumes all points have the same dimension
    # Select axis based on depth so that axis cycles through all valid values
    axis = depth % k


    # Sort point list by axis and choose median as pivot element
    point_list.sort(key=itemgetter(axis))
    median = len(point_list) // 2


    # Create node and construct subtrees
    return Node(
        location=point_list[median],
        left_child=kdtree(point_list[:median], depth + 1),
        right_child=kdtree(point_list[median + 1 :], depth + 1),
    )


def main():
    """Example usage"""
    point_list = [(7, 2), (5, 4), (9, 6), (4, 7), (8, 1), (2, 3)]
    tree = kdtree(point_list)
    print(tree)


if __name__ == "__main__":
    main()
```
Curse of dimensionality:
- Each split halves the training data size
- Pretty soon, run of data.
  - As a general rule, if the dimensionality is $k$, the number of points in the data, $n$, should be $2^k$.
    - So, theoretically,  20 attributes needs a million rows. 
    - Strange... we often achieve competency on much smaller data sets.


### Recursive Fastmap (the sampling way)
<!-- <img align=right src="/etc/img/pca.png" width=500> -->


"Fastmap" belongs to a class of approximation algorithms to principal component analysis (PCA).
- Eigenvectors of the correlation matrix. 
- e.g. at image on right, what is the dimension that best models the data?
  - maybe not the $x,y$ dimensions but the direction of that big arrow
    - which points along the direction of greatest variance.
      - and the 2nd component is the little arrow at right-angles to the first principal arrow.


<br clear=all>
<!-- <img align=right src="/etc/img/abc.png"> -->
We can simplify PCA from polynominal to near-linear time with FASTMAP.
Once we know distance, then we  project things in $N$ dimensions down to one dimension 
(being a line between 2 distant points).

```lua
function cosine(a,b,c,    x1,x2,y) 
  x1 = (a^2 + c^2 - b^2) / (2*c)
  x2 = math.max(0, math.min(1, x1)) 
  y  = (a^2 - x2^2)^.5
  return x2, y end
```


Project every point to a line connecting two distance items.
```lua
function DATA.half(i,rows,  cols,above) --> t,t,row,row,row,n; divides data using 2 far points
  local A,B,left,right,c,dist,mid,some,project
  function project(row)    return {row=row, dist=cosine(dist(row,A), dist(row,B), c)} end
  function dist(row1,row2) return i:dist(row1,row2,cols) end
  rows = rows or i.rows
  some = many(rows,the.Sample)
  A    = above or any(some)  -- if a parent found a distant point, use that
  B    = i:around(A,some)[(the.Far * #rows)//1].row
  c    = dist(A,B)
  left, right = {}, {}
  for n,tmp in pairs(sort(map(rows, project), lt"dist")) do
    if   n <= #rows//2 
    then push(left,  tmp.row); mid = tmp.row
    else push(right, tmp.row) end end
  return left, right, A, B, mid, c end
```
Once we can divide some data in two, then recursive clustering is just recursive division.


```lua
function DATA.cluster(i,  rows,min,cols,above) --> t; returns `rows`, recursively halved
  local node,left,right,A,B,mid
  rows = rows or i.rows
  min  = min or (#rows)^the.min
  cols = cols or i.cols.x
  node = {data=i:clone(rows)} --xxx cloning
  if #rows > 2*min then
    left, right, node.A, node.B, node.mid = i:half(rows,cols,above)
    node.left  = i:cluster(left,  min, cols, node.A)
    node.right = i:cluster(right, min, cols, node.B) end
  return node end


```
Which, just to remind us, gives us this:
```
398  {:Acc+ 15.6 :Lbs- 2970.4 :Mpg+ 23.8}
| 199
| | 99
| | | 49
| | | | 24  {:Acc+ 17.3 :Lbs- 2623.5 :Mpg+ 30.4}
| | | | 25  {:Acc+ 16.3 :Lbs- 2693.4 :Mpg+ 29.2}
| | | 50
| | | | 25  {:Acc+ 15.8 :Lbs- 2446.1 :Mpg+ 27.2}
| | | | 25  {:Acc+ 16.7 :Lbs- 2309.2 :Mpg+ 26.0}
| | 100
| | | 50
| | | | 25  {:Acc+ 16.2 :Lbs- 2362.5 :Mpg+ 32.0}
| | | | 25  {:Acc+ 16.4 :Lbs- 2184.1 :Mpg+ 34.8}
| | | 50
| | | | 25  {:Acc+ 16.2 :Lbs- 2185.8 :Mpg+ 29.6}
| | | | 25  {:Acc+ 16.3 :Lbs- 2179.4 :Mpg+ 26.4}
| 199
| | 99
| | | 49
| | | | 24  {:Acc+ 16.6 :Lbs- 2716.9 :Mpg+ 22.5}
| | | | 25  {:Acc+ 16.1 :Lbs- 3063.5 :Mpg+ 20.4}
| | | 50
| | | | 25  {:Acc+ 17.4 :Lbs- 3104.6 :Mpg+ 21.6}
| | | | 25  {:Acc+ 16.3 :Lbs- 3145.6 :Mpg+ 22.0}
| | 100
| | | 50
| | | | 25  {:Acc+ 12.4 :Lbs- 4320.5 :Mpg+ 12.4}
| | | | 25  {:Acc+ 11.3 :Lbs- 4194.2 :Mpg+ 12.8}
| | | 50
| | | | 25  {:Acc+ 13.7 :Lbs- 4143.1 :Mpg+ 18.0}
| | | | 25  {:Acc+ 14.4 :Lbs- 3830.2 :Mpg+ 16.4}
```

### Random Projections


![](https://ars.els-cdn.com/content/image/1-s2.0-S0031320315003945-gr2.jpg)


- Method one: Guassian random projections
   - Matrix = rows \* cols
   - Matrix A,B
   - A =        m × n 
   - B =        n × p 
   - C = A\*B = m x  p
   - So we can reduce n=1000 cols in matrix A to p=30 cols in C via a matrix
      - 1000 row by 30 cols
   - Initialize B by filling each column with a random number pulled from a normal bell curve
   - Only works if all the data is numeric
   - Requires all the data in RAM (bad for big data sets)
- Method two:  LSH: locality sensitivity hashing
   - Find a way to get an approximate position for a row, in a reduced space
      - e.g. repeat d times
          -  Find  two  distant (\*) "pivots"  = (p,o)
              - Pick, say, 30 rows at random then find within them, the most distant rows
          - the i-th dimension:
              - this row's d.i = 0 if (row closer to p than o) else 1
      - repeat d=log(sqrt(N))+1 pivots to get  d random projectsion
   - If you want not 0,1 on each dimension but a continuous number then:
      - given pivots (A,B) separated by "c"
      - a = dist(row,A)
      - b = dist(row,B)
      - this row's d.x = (a^2 + c^2 - b^2) / (2c)
          - Cosine rule

