# Evosuite tutorial

Part of it is taken from the official [website](https://www.evosuite.org/documentation/tutorial-part-1/).

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
docker run --rm -v $PWD/Tutorial_Stack:/home -v $PWD/.m2_local:/root/.m2 -w /home maven:3.3.9 mvn compile
```

## Run Evosuite

```bash
docker run --rm -u $UID -v $PWD/Tutorial_Stack:/evosuite evosuite/evosuite -class tutorial.Stack -projectCP target/classes -criterion branch
```

Finishes quickly as there are only 7 branches (one per method and two per if statement) to cover and all of them are feasible. Now run Evosuite on the `StackNew` class:

```bash
docker run --rm -u $UID -v $PWD/Tutorial_Stack:/evosuite evosuite/evosuite -class tutorial.StackNew -projectCP target/classes -criterion branch
```

The computation goes ahead until the budget expires (1 minute), without covering all the branches (9 in this case, i.e., 1 per method, also the private one, and 2 per if statement). There is one infeasible branch in the `push` method:

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





