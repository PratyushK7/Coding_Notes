# Sorting

> **Selection Sort:** Unstable \(Can be turned stable\), Ω\(N²\) - θ\(N²\) - O\(N²\) : O\(1\)  
> **Bubble Sort:** Stable**,** Ω\(N\) - θ\(N²\) - O\(N²\) : O\(1\)  
> **Insertion Sort:** Stable, Ω\(N\) - θ\(N²\) - O\(N²\) : O\(1\)  
> **Merge Sort:** Stable, Ω\(NlogN\) - θ\(NlogN\) - O\(NlogN\) : O\(N\)  
> **Quick Sort:** Unstable \(Can be turned stable\), Ω\(NlogN\) - θ\(NlogN\) - O\(N²\) : O\(1\)

> **Insertion sort** is **faster** for **small** n because Quick **Sort** has extra overhead from the recursive function calls. Due to recursion constant factor is higher. **Insertion sort** requires less memory then quick sort \(stack memory\).

```java
// insertion sort
    static void insertionSort(int[] arr) {
        int n = arr.length;
        for (int i = 1; i < n; i++) {
            // System.out.println(Arrays.toString(arr));
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }

    // merge sort
    static void mergeSort(int[] arr, int l, int r) {
        if (l < r) {
            int m = (l + r) / 2;
            mergeSort(arr, l, m);
            mergeSort(arr, m + 1, r);
            merge(arr, l, m, r);
        }
    }

    static void merge(int[] arr, int l, int m, int r) {
        int[] res = new int[arr.length];
        int n1 = m - l + 1, n2 = r - m;

        int[] L = new int[n1];
        for (int i = 0; i < n1; i++) {
            L[i] = arr[l + i];
        }
        int[] R = new int[n2];
        for (int i = 0; i < n2; i++) {
            R[i] = arr[m + 1 + i];
        }
        int i = 0, j = 0, k = l;

        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) {
                arr[k++] = L[i++];
            } else {
                arr[k++] = R[j++];
            }
        }

        while (i < n1) {
            arr[k++] = L[i++];
        }

        while (j < n2) {
            arr[k++] = R[j++];
        }

    }

```

> **Time Complexity:** Time complexity of heapify is O(Logn). Time complexity of createAndBuildHeap() is O(n) and overall time complexity of Heap Sort is O(nLogn).
>
> **Applications of HeapSort** 
> **1.** [Sort a nearly sorted (or K sorted) array](https://www.geeksforgeeks.org/nearly-sorted-algorithm/) 
> **2.** [k largest(or smallest) elements in an array](https://www.geeksforgeeks.org/k-largestor-smallest-elements-in-an-array/)

```cpp
/* Heap is in array form, for all leaves i.e. from i = 0 to n/2
in reverse order heapify is called, child then root otherwise
child will cause unnecessary changes. After getting proper heap,
it's 0th element gives max (or min in min heap) take that and
put it at the end now ignore thatlast value because it's sorted
and again apply heapify on theremaining.*/
void heapify(int arr[], int n, int i)    // max heap here
{
    int parent = i, left = 2 * i + 1, right = 2 * i + 2;
    if (left < n && arr[parent] < arr[left])
    {
        swap(arr[left], arr[parent]);
        heapify(arr, n, left);
    }
    else if (right < n && arr[parent] < arr[right])
    {
        swap(arr[right], arr[parent]);
        heapify(arr, n, right);
    }
}
void heapSort(int arr[], int n)
{
    // Building heap part is O(N)
    // https://www.geeksforgeeks.org/time-complexity-of-building-a-heap/
    for (int i = n / 2 - 1; i >= 0; --i)
        heapify(arr, n, i);
    // ascending order sort
    for (int i = n - 1; i >= 0; --i)
    {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}
```

## Radix Sort

[https://www.youtube.com/watch?v=JMlYkE8hGJM](https://www.youtube.com/watch?v=JMlYkE8hGJM)

https://www.youtube.com/watch?v=Il45xNUHGp0

O\(d \* \(n + b\)\) - b is base in number it is 10, for strings it's 26

* Most optimal, only drawback is due to not being comparison based sorting in nature only limited to int, float or strings.

* > The [lower bound for Comparison based sorting algorithm](https://www.geeksforgeeks.org/lower-bound-on-comparison-based-sorting-algorithms/) (Merge Sort, Heap Sort, Quick-Sort .. etc) is Ω(nLogn), i.e., they cannot do better than nLogn. 
  >
  > [Counting sort](https://www.geeksforgeeks.org/counting-sort/) is a linear time sorting algorithm that sort in O(n+k) time when elements are in the range from 1 to k.
  >
  > ***What if the elements are in*** **the** ***range from 1 to n2?*** 
  > We can’t use counting sort because counting sort will take O(n2) which is worse than comparison-based sorting algorithms. Can we sort such an array in linear time? 
  >
  > [Radix Sort](http://en.wikipedia.org/wiki/Radix_sort) is the answer. The idea of Radix Sort is to do digit by digit sort starting from least significant digit to most significant digit. Radix sort uses counting sort as a subroutine to sort.

```java
    // radix sort
    static void radixSort(int[] arr, int n) {
        int max = getMax(arr, n);
        for (int exp = 1; max / exp > 0; exp *= 10) {
            countSort(arr, n, exp);
        }
    }

    static int getMax(int[] arr, int n) {
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            max = Math.max(max, arr[i]);
        }
        return max;
    }

    static void countSort(int[] arr, int n, int exp) {
        int[] output = new int[n];
        int[] count = new int[10];
        Arrays.fill(count, 0);

        int i;
        for (i = 0; i < n; i++) {
            count[(arr[i] / exp) % 10]++;
        }

        // update count array such that it will contain the actual position of the
        // element
        for (i = 1; i < n; i++) {
            count[i] += count[i - 1];
        }

        // build output array starting from last: to maintain the stability of count
        // array(counting sort property)
        for (i = n - 1; i >= 0; i--) {
            output[count[(arr[i] / exp) % 10] - 1] = arr[i];
            count[(arr[i] / exp) % 10]--;
        }

        for (i = 0; i < n; i++) {
            arr[i] = output[i];
        }
    }
```

## Partial Sorting

Time complexity: O\(n + klogk\)

```cpp
// 10, 45, 60, 78, 23, 21, 3
partial_sort(vec.begin(), vec.begin() + 3, vec.end());
// gives 3 10 21 78 60 45 23
/* If say problem was add k elements from array in sorted order then doing it
with a priority queue is same as sorting all at once O(nlogn) using partial
sort is better */

// stl uses heap sort internally
```

## Kth Smallest/Largest Number

> **Method 4 (QuickSelect)**
> This is an optimization over method 1 if [QuickSort ](http://geeksquiz.com/quick-sort/)is used as a sorting algorithm in first step. In QuickSort, we pick a pivot element, then move the pivot element to its correct position and partition the array around it. The idea is, not to do complete quicksort, but stop at the point where pivot itself is k’th smallest element. Also, not to recur for both left and right sides of pivot, but recur for one of them according to the position of pivot. The worst case time complexity of this method is O(n2), but it works in O(n) on average.
>
> **https://www.geeksforgeeks.org/kth-smallestlargest-element-unsorted-array/**

```java
    // kth smallest/ largest
    // Quick sort extension
    static int kthSmallest(int[] arr, int l, int r, int k) {
        if (k <= 0 || k > r - l + 1)
            return Integer.MAX_VALUE;
        int pos = partition(arr, l, r);
        if (pos - l == k - 1)
            return arr[pos];
        if (pos - l > k - 1)
            return kthSmallest(arr, l, pos - 1, k);
        return kthSmallest(arr, pos + 1, r, k - (pos - l + 1));
    }

    static int partition(int[] arr, int l, int r) {
        int pivot = arr[r];
        int i = l, j = l;
        while (i <= r) {
            if (arr[i] <= pivot) {
                swap(arr, i, j);
                j++;
            }
            i++;
        }
        return j - 1;
    }
```

## Count Inversions

```java
// BruteFoce O(N²)
int invCount(vector<int> &arr)
{
    int inv_count = 0;
    for (int i = 0; i < arr.size(); ++i)
    {
        for (int j = i + 1; j < arr.size(); ++j)
            if (arr[i] > arr[j]) inv_count++;
    }
    return inv_count;
}

/* Merge Sort extension O(nlogn)
Whenever there's a swap while merging, there will be m-i+1 inversions */
/*MERGE SORT EXTENSION (nlogn)
 Create a function merge that counts the number of inversions when two halves
 of the array are merged, create two indices i and j, i is the index for first
 half and j is an index of the second half. if a[i] is greater than a[j], then
 there are (mid – i) inversions. because left and right subarrays are sorted,
 so all the remaining elements in left-subarray (a[i+1], a[i+2] … a[mid]) will
 be greater than a[j]. */

    static int mergeSortAndCount(int[] arr, int l, int r) {
        int count = 0;

        if (l < r) {
            int mid = (l + r) / 2;
            count += mergeSortAndCount(arr, l, mid);
            count += mergeSortAndCount(arr, mid + 1, r);
            count += mergeAndCount(arr, l, mid, r);
        }

        return count;
    }

    static int mergeAndCount(int[] arr, int l, int m, int r) {
        int[] temp = Arrays.copyOf(arr, arr.length);
        
        int i = l, j = m + 1, k = l, inv_count = 0;
        while (i <= m && j <= r) {
            if (temp[i] > temp[j]) {
                arr[k++] = temp[j++];
                inv_count += m - i + 1;
            } else {
                arr[k++] = temp[i++];
            }
        }
        while (i <= m)
            arr[k++] = temp[i++];

        while (j <= r)
            arr[k++] = temp[j++];

        return inv_count;
    }


/* Using Fenwick tree O(nlogn)
For an element x inside array we need to find how many elements are smaller
that it and appeared before. Initialize a BIT with all zero once element x
is visited update it's count +1 in BIT. Find query for element >= x
thats inv count */
int invCount(vector<int> &arr)
{
    int inv_count = 0;
    for (auto &x : arr)
    {
        inv_count += query(MAXN) - query(x);
        update(x, 1);
    }
}
```

## Minimum Number of Swaps required to Sort array

`So at each i starting from 0 to N in the given array, where N is the size of the array:`

`1. If i is not in its correct position according to the sorted array, then`

`2. We will fill this position with the correct element from the hashmap we built earlier. We know the correct element which should come here is temp[i], so we look up the index of this element from the hashmap.` 

`3. After swapping the required elements, we update the content of the hashmap accordingly, as temp[i] to the ith position, and arr[i] to where temp[i] was earlier.`

```java
// MINIMUM SWAPS REQUIRED TO SORT ARRAY
static int minSwaps(int[] arr) {
    int n = arr.length;

    HashMap<Integer, Integer> hm = new HashMap<>();
    for (int i = 0; i < n; i++) {
        hm.put(arr[i], i);
    }

    int[] temp = Arrays.copyOf(arr, n);
    Arrays.sort(temp);

    int swaps = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] != temp[i]) {
            swaps++;
            int initial = arr[i];
            swap(arr, i, hm.get(temp[i]));
            hm.put(initial, hm.get(temp[i]));
            hm.put(temp[i], i);
        }
    }

    // System.out.println(swaps);
    return swaps;
}

```

## [Merge Intervals](https://leetcode.com/problems/merge-intervals)

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        int n = intervals.length;
        if(n == 1) return intervals;
        
        Arrays.sort(intervals, new Comparator<int[]>(){
            public int compare(int[] a, int[] b){
                return a[0]-b[0];
            }
        });    
        
        ArrayList<int[]> al = new ArrayList<>();
        int s = intervals[0][0];
        int e = intervals[0][1];
        int i = 1;
        while(i < n){
            int[] a = intervals[i];
            if(e < a[0]){
                al.add(new int[]{s, e});
                s = a[0];
                e = a[1];
            }else if(e >= a[0] && e < a[1]){
                e = a[1];
            }  
            i++;
        }
        al.add(new int[]{s, e});
    
        int[][] arr = new int[al.size()][2];
        
        for(int j=0; j<arr.length; j++){
            int[] b = al.get(j);
            arr[j][0] = b[0];
            arr[j][1] = b[1];
        }
        
        return arr;
    }
}
```

## [Insert Interval In a Sorted List](https://leetcode.com/problems/insert-interval/)

In 2 pass found start and end bound. Start is max index with intervals\[start\]\[1\] &lt;= newInterval\[0\] and end is min index with intervals\[end\]\[0\] &lt;= newInterval\[1\]

```cpp
class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval)
    {
        if (intervals.empty()) return {newInterval};
        int l = 0, r = intervals.size()-1;
        while (l <= r)
        {
            int mid = (l+r)/2;
            if (intervals[mid][1] == newInterval[0]) { l = mid; break; }
            if (intervals[mid][1] < newInterval[0]) l = mid+1;
            else r = mid-1;
        }
        int start = l; r = intervals.size()-1;
        while (l <= r)
        {
            int mid = (l+r)/2;
            if (intervals[mid][0] == newInterval[1]) { r = mid; break; }
            if (intervals[mid][0] < newInterval[1]) l = mid+1;
            else r = mid-1;
        }
        int end = r;
        if (start < intervals.size())
            newInterval[0] = min(newInterval[0], intervals[start][0]);
        if (end >= 0)
            newInterval[1] = max(newInterval[1], intervals[end][1]);
        intervals.erase(intervals.begin()+start, intervals.begin()+end+1);
        intervals.insert(intervals.begin()+start, newInterval);
        return intervals;
    }
};
```

## [Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/)

```cpp
vector<vector<int>> intervalIntersection(vector<vector<int>>& A, vector<vector<int>>& B)
{
    vector<vector<int>> res;
    for (int i = 0, j = 0; i < A.size() && j < B.size();)
    {
        int p = max(A[i][0], B[j][0]);
        int q = min(A[i][1], B[j][1]);
        if (p <= q)
        {
            res.push_back({p, q});
            if (q == A[i][1]) ++i;
            else ++j;
        }
        else
        {
            if (A[i][0] > B[j][1]) ++j;
            else ++i;
        }
    }
    return res;
}
```

## [Hotel Booking Problem](https://www.interviewbit.com/problems/hotel-bookings-possible/)

```cpp
/* Consider it like a timeline +1 means arrival -1 means departure then sort it
and add to find rooms required at an instance. */
bool Solution::hotel(vector<int> &arrive, vector<int> &depart, int K)
{
    vector<pair<int, int>> v;
    int n = arrive.size();
    for (int i = 0; i < n; ++i)
    {
        v.push_back({arrive[i], 1});
        v.push_back({depart[i], -1});
    }
    sort(v.begin(), v.end());
    int count = 0;
    for (auto &x : v)
    {
        count += x.second;
        if (count > k) return false;
    }
    return true;
}
```

## [Largest Number](https://www.interviewbit.com/problems/largest-number/)

```cpp
string Solution::largestNumber(const vector<int> &A)
{
    vector<string> b;
    for (auto &x : A) b.push_back(to_string(x));
    sort(b.begin(), b.end(), [](string x, string y)
    {
        string xy = x.append(y);
        string yx = y.append(x);
        return xy.compare(yx) > 0 ? true : false;
    });
    string ans = "";
    for (int i = 0; i < b.size(); i++) ans += b[i];
    bool allzero = true;
    for (char i : ans)
        if (i != '0') { allzero = false; break; }
    if (allzero) return "0";
    return ans;
}
```

## [Max Distance](https://www.interviewbit.com/problems/max-distance/)

```cpp
int Solution::maximumGap(const vector<int> &A)
{
    int n = A.size();
    if (n == 0) return -1;
    vector<pair<int, int>> arr;
    for (int i = 0; i < n; ++i) arr.push_back({A[i], i});
    sort(arr.begin(), arr.end());
    int ans = 0, rmax = arr[n - 1].second;
    for (int i = n - 2; i >= 0; --i)
    {
        ans = max(ans, rmax - arr[i].second);
        rmax = max(rmax, arr[i].second);
    }
    return ans;
}
```

## Missing Integer in an array

```cpp
/* If the array contains all numbers from 1 to N except one of them is missing
(n*(n+1))/2 - sum_of_array is answer */

/* First missing integer
Mark every element visited as negative within array only */
for (auto it = A.begin(); it != A.end();) // first remove negative numbers
{
    if (*it <= 0) A.erase(it);
    else ++it;
}
for (auto &x : A)
{
    int cur = abs(x)-1;
    if (cur >= 0 && cur < A.size()) A[cur] = -A[cur];
}
for (int i = 0; i < A.size(); i++)
    if (A[i] > 0) return i + 1;
return A.size()+1;
```

## [Missing Ranges](https://www.lintcode.com/problem/missing-ranges/description)
```c++
vector<string> findMissingRanges(vector<int> &nums, int lower, int upper)
{
    vector<string> res;
    long long l = lower, r = upper;
    for (const int x : nums)
    {
        if (x > l)
        {
            if (l == x-1) res.push_back(to_string(l));
            else res.push_back(to_string(l) + "->" + to_string(x-1));
        }
        l = (long long)x + 1;
    }
    if (l <= r)
    {
        if (l == r) res.push_back(to_string(l));
        else res.push_back(to_string(l) + "->" + to_string(r));
    }
    return res;
}
```

## Maximum Consecutive Gap

```cpp
/* Basically if we could have sort then it's simple how about counting sort?
It would have worked if number range was low.
We have to reduce comparisons in sorting, we can use idea of bucket sort.
Considering uniform gap:
min, min + gap, min + 2*gap, ... min + (n-1)*gap
min + (n-1)*gap = max
gap = (max - min) / (n-1)
Now we place every number in these n-1 buckets:
bucket = (num - min)/gap, we prepare max and min for each bucket and subtract
to find our answer */
int Solution::maximumGap(const vector<int> &A)
{
    int n = A.size();
    if (n < 2) return 0;
    vector<int> forMin(n, INT_MAX), forMax(n, INT_MIN);
    int mn = *min_element(A.begin(), A.end());
    int mx = *max_element(A.begin(), A.end());
    int gap = (mx-mn) / (n-1);
    if (gap == 0) return (mx-mn);
    for (auto &x : A)
    {
        int bucket = (x - mn) / gap;
        forMin[bucket] = min(x, forMin[bucket]);
        forMax[bucket] = max(x, forMax[bucket]);
    }
    int maxDiff = 0;
    for (int i = 0, j = 0; i < forMin.size(); ++i)
    {
        if (forMin[i] >= 0)
        {
            maxDiff = max(maxDiff, forMin[i] - forMax[j]);
            j = i;
        }
    }
    return maxDiff;
}
```

## Min/Max XOR Pair

```cpp
/* If we sort and then check xor of consecutive pair it will give min
no need to check pair which are not consecutive. */
int Solution::findMinXor(vector<int> &A)
{
    sort(A.begin(), A.end());
    int mn = INT_MAX;
    for (int i = 0; i < A.size()-1; ++i)
        mn = min(ans, A[i]^A[i+1]);
    return mn;
}
// Only work for min XOR
```

## `Can be solved using trie along with Max XOR Pair problem`

```cpp
/* No need to check for isTerminal since we have same length
(i.e. 32) for all */
vector<vector<int>> trieArr;
int nxt;
void insert(int num)
{
    int pos = 0;
    for (int i = 31; i >= 0; --i)
    {
        bool cur = !!(num & (1<<i));
        if (trieArr[pos][cur] == 0)
        {
            trieArr[pos][cur] = nxt;
            pos = nxt++;
        }
        else pos = trieArr[pos][cur];
    }
}
int findOptimal(int num)
{
    int pos = 0, res = 0;
    for (int i = 31; i >= 0; --i)
    {
        bool cur = !!(num & (1<<i));
        if (trieArr[pos][!cur] != 0)
        {
            pos = trieArr[pos][!cur];
            if (!cur) res += (1<<i);
        }
        else
        {
            pos = trieArr[pos][cur];
            if (cur) res += (1<<i);
        }
    }
    return res;
}
int findMaximumXOR(vector<int>& nums)
{
    trieArr.assign((nums.size()+1)*32, vector<int>(2, 0));
    nxt = 1;
    
    for (auto &x : nums) insert(x);
    int res = 0;
    for (auto &x : nums) res = max(res, x^findOptimal(x));
    return res;
}
```

## [The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/)

![\[ \[2 9 10\], \[3 7 15\], \[5 12 12\], \[15 20 10\], \[19 24 8\] \] -&amp;gt; \[ \[2 10\], \[3 15\]...](.gitbook/assets/image%20%28264%29.png)

Can also be solved using segment tree

```cpp
class Solution {
public:
    vector<vector<int>> getSkyline(vector<vector<int>>& buildings)
    {
        vector<pair<int, int>> edges;
        for (auto &x : buildings)
        {
            edges.push_back({x[1], x[2]});
            edges.push_back({x[0], -x[2]});
        }
        sort(edges.begin(), edges.end());
        
        /*
            [9, 10]                    [2, -10]
            [2, -10]                   [3, -15]
            [7, 15]                    [5, -12]
            [3, -15]                   [7, 15]
            [12, 12]        ->         [9, 10]
            [5, -12]                   [12, 12]
            [20, 10]                   [15, -10]
            [15, -10]                  [19, -8]
            [24, 8]                    [20, 10]
            [19, -8]                   [24, 8]
        */

        multiset<int> st;
        st.insert(0);
        int cur = 0;
        vector<vector<int>> res;
        for (auto &x : edges)
        {
            if (x.second < 0) st.insert(-x.second); // -ve is starting
            else st.erase(st.find(x.second));
            if (cur != *st.rbegin())
            {
                res.push_back({x.first, *st.rbegin()});
                cur = *st.rbegin();
            }
        }
        return res;
    }
};
```

