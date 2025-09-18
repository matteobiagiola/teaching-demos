# Evosuite tutorial

https://www.evosuite.org/documentation/tutorial-part-1/

## Setup env

```bash
docker pull evosuite/evosuite
docker pull maven:3.3.9
```

## Download stack tutorial project

```bash
wget http://evosuite.org/files/tutorial/Tutorial_Stack.zip
unzip Tutorial_Stack.zip
```

## Compile project

```bash
docker run -v $(pwd)/Tutorial_Stack:/home -w /home maven:3.3.9 mvn compile
```

## Run Evosuite

```bash
docker run -it -u ${UID} -v ${PWD}/Tutorial_Stack:/evosuite evosuite/evosuite -class tutorial.Stack -projectCP target/classes -criterion branch
```

Finishes quickly as there are only 7 branches to cover and all of them are feasible. Now create a new file named `StackNew.java` with the following content:

```java
package tutorial;

import java.util.EmptyStackException;

public class StackNew {
    private int[] values = new int[3];
    private int size = 0;

    void push(int x) {
        if (size >= values.length) {
            resize();
        }
        if (size < values.length) {
            values[size++] = x;
        }
    }

    int pop() {
        if (size > 0) {
            return values[--size];
        } else {
            throw new EmptyStackException();
        }
    }

    private void resize() {
        int[] tmp = new int[values.length * 2];
        for (int i = 0; i < values.length; i++) {
            tmp[i] = values[i];
        }
        values = tmp;
    }
}
```

and compile it:

```bash
docker run -v $(pwd)/Tutorial_Stack:/home -w /home maven:3.3.9 mvn compile
```

then run Evosuite:

```bash
docker run -it -u ${UID} -v ${PWD}/Tutorial_Stack:/evosuite evosuite/evosuite -class tutorial.StackNew -projectCP target/classes -criterion branch
```

The computation goes ahead until the budget expires (1 minute), without covering all the branches. There is one infeasible branch in the `push` method:

```java
package tutorial;

import java.util.EmptyStackException;

public class StackNew {
    private int[] values = new int[3];
    private int size = 0;

    void push(int x) {
        if (size >= values.length) {
            resize();
        }
        if (size < values.length) { // <-- else branch infeasible
            values[size++] = x;
        }
    }
    ...
}
```





