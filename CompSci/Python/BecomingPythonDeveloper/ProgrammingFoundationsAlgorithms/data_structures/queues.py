from collections import deque

queue = deque()

queue.append(1)
queue.append(2)
queue.append(3)
queue.append(4)

print(queue)

x = queue.popleft()
print(queue)
print(x)
