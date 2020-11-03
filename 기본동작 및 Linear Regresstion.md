# Tensorflow 기본 구성이 어떻게 되는지 및 동작 방법

- Tensorflow 기본 설계 및 동작 순서
    1. 노드를 설정해준다 (변수들의 값, 수식 값 등등)
    2. 노드들을 묶어 준다(변수들의 값과 수식의 값을 연결)
    3. sess = tensorflow.Session() 함수를 통하여서 세션 생성해준다
    4. sess.run() 함수를 통하여서 세션 실행
    ```python
import tensorflow as tf

# 노드 설정
x = tf.constant("Hello")

# 세션 설정
sess = tf.Session()

# 세션 실행
print(sess.run(x)) # Hello가 출력된다.
```

- Tensorflow로 간단한 Linear Regression 구현하기
    - cost function(loss function) ⇒ H(x) = Wx + b

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b701c1db-7664-4806-b7b2-a7d8f1bb9e53/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b701c1db-7664-4806-b7b2-a7d8f1bb9e53/Untitled.png)

    ```python
    import tensorflow as tf

    # 노드 생성
    x = [1,2,3]
    y = [3,6,9]

    w = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = w*x + b

    # cost function(loss function)
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train = optimizer.minimizer(cost)

    # 세션 생성 및 실행
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 학습시키기
    for step in range(2001):
    		sess.run(train)
    		if step%20 == 0:
    				print(step, sess.run(cost), sess.run(w), sess.run(b))
    ```

    ```python
    import tensorflow as tf

    # 노드 생성
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = w*x + b

    # cost function(loss function)
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train = optimizer.minimizer(cost)

    # 세션 생성 및 실행
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 학습시키기
    for step in range(2001):
    		cost_val, w_val, b_val = sess.run([cost, w, b, train], feed_dict={x:[1,2,3], y:[2,4,6]})
    		if step%20 == 0:
    				print(step, sess.run(cost), sess.run(w), sess.run(b))
    ```
