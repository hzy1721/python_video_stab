from collections import deque


class PopDeque(deque):
    # 能够返回被弹出元素的双端队列

    def deque_full(self):
        """Test if queue is full"""
        return len(self) == self.maxlen

    def pop_append(self, x):
        """deque.append helper to return popped element if deque is at ``maxlen``

        :param x: element to append
        :return: result of ``deque.popleft()`` if deque is full; else ``None``

        >>> x = PopDeque([0], maxlen=2)
        >>> x.pop_append(1)

        >>> x.pop_append(2)
        0
        """
        # 被弹出的元素
        popped_element = None
        # 队列已满
        if self.deque_full():
            # 从队头弹出一个元素，存放到变量中
            popped_element = self.popleft()

        # 队尾添加一个元素
        self.append(x)

        # 返回可能的被弹出元素(可能是None)
        return popped_element

    def increment_append(self, increment=1, pop_append=True):
        """Append deque[-1] + ``increment`` to end of deque

        If deque is empty then 0 is appended

        :param increment: size of increment in deque
        :param pop_append: return popped element if append forces pop?
        :return: popped_element if pop_append is True; else None
        """
        # 队列为空
        if len(self) == 0:
            # 入队0
            popped_element = self.pop_append(0)
        # 队列不为空
        else:
            # 把队尾最后一个元素加上increment再入队
            popped_element = self.pop_append(self[-1] + increment)

        # 不需要返回弹出元素
        if not pop_append:
            return None

        # 返回弹出元素
        return popped_element
