## Custom sort

```java
Arrays.sort(arr, new Comparator<int[]>(){
	public int compare(int[] a, int[] b){
		return a[0]-b[0];
	}
})
```

## Sort Priority Queue

```java
PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0]-b[0]);
```

