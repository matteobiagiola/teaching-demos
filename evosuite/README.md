# Evosuite tutorial

Part of it is taken from the official [website](https://www.evosuite.org/documentation/tutorial-part-1/).

## Setup env

```bash
docker pull evosuite/evosuite
docker pull maven:3.3.9
```

## Compile project

```bash
docker run --rm -v $PWD/Tutorial_Stack:/home -v $PWD/.m2_local:/root/.m2 -w /home maven:3.3.9 mvn compile
```

## Run Evosuite

```bash
docker run --rm -u $UID -v $PWD/Tutorial_Stack:/evosuite evosuite/evosuite -class tutorial.Stack -projectCP target/classes -criterion branch
```

### Branch coverage intuition

Let us suppose we have two tests:

```java
// Test A
Stack<Integer> stack0 = new Stack<Integer>();
Integer integer0 = new Integer(1);
stack0.push(integer0);
stack0.push(integer0);
stack0.push(integer0);
stack0.push(integer0);
```

```java
// Test B
Stack<Integer> stack0 = new Stack<Integer>();
Integer integer0 = new Integer(1);
stack0.push(integer0);
stack0.push(integer0);
stack0.push(integer0);
stack0.push(integer0);
stack0.push(integer0);
```

Test B is _closer_ to satisfy the true branch of of the `push` method (as the difference between `capacity` and `pointer` is smaller).

Finishes quickly as there are only 7 branches (one for the constructor and two per if statement) to cover and all of them are feasible. The output is stored in `evosuite-tests`, with the `Stack_ESTEst.java` containing the test suite. In this case, evosuite generates 5 tests to cover all the 7 branches. The `evosuite-report` directory stores some overall statistics abot the run, and the `covered.goals` file that shows what each of the 5 tests covers; the number of goals is greater than 7 because some tests cover the same goals. The goals covered by each test are also reported as a comment above each test in the `Stack_ESTEst.java` class. The goals covered are:

```
test0,tutorial.Stack.<init>()V: root-Branch
test1,tutorial.Stack.push(Ljava/lang/Object;)V: I6 Branch 1 IF_ICMPLT L12 - true
test4,tutorial.Stack.push(Ljava/lang/Object;)V: I6 Branch 1 IF_ICMPLT L12  - false
test3,tutorial.Stack.pop()Ljava/lang/Object;: I4 Branch 2 IFGT L18 - true
test2,tutorial.Stack.pop()Ljava/lang/Object;: I4 Branch 2 IFGT L18 - false
test0,tutorial.Stack.isEmpty()Z: I4 Branch 3 IFGT L24 - false
test1,tutorial.Stack.isEmpty()Z: I4 Branch 3 IFGT L24 - true
```

The branches do not correspond the branches in the source code, as instrumentation is carried out at the bytecode level.
In particular, `javac` usually inverts conditions (`<` becomes `IF_ICMPGE`, `==` becomes `IF_ICMPNE`, and so on), so for most if statements, EvoSuite's "true" is the source's "false").

Now, let's take another example, i.e., the `StackNew` class and run Evosuite on it:

```bash
docker run --rm -u $UID -v $PWD/Tutorial_Stack:/evosuite evosuite/evosuite -class tutorial.StackNew -projectCP target/classes -criterion branch
```

The computation goes ahead until the budget expires (1 minute), without covering all the branches (9 in this case, i.e., 1 is the constructor, and 2 per if statement / for statement, also the private one). There is one infeasible branch in the `push` method:

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

The goals covered are:
```
test0,tutorial.StackNew.<init>()V: root-Branch
test0,tutorial.StackNew.push(I)V: I7 Branch 1 IF_ICMPLT L10 - true
test2,tutorial.StackNew.push(I)V: I7 Branch 1 IF_ICMPLT L10 - false
test0,tutorial.StackNew.push(I)V: I19 Branch 2 IF_ICMPGE L13 - false
test1,tutorial.StackNew.pop()I: I4 Branch 3 IFLE L19 - true
test0,tutorial.StackNew.pop()I: I4 Branch 3 IFLE L19 - false
test2,tutorial.StackNew.resize()V: I18 Branch 4 IF_ICMPGE L28 - true
test2,tutorial.StackNew.resize()V: I18 Branch 4 IF_ICMPGE L28 - false
```

The missed coverage goal is printed at the end of the generation:

```
* Coverage analysis for criterion BRANCH
 - Missed goal tutorial.StackNew.push(I)V: I19 Branch 2 IF_ICMPGE L13 - true
```







