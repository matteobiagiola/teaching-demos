# PIT tutorial

Download the maven container image:

```bash
docker pull maven:3.3.9
```

Compile the tests:

```bash
docker run --rm -v $PWD:/home -v $PWD/.m2_local:/root/.m2 -w /home maven:3.3.9 mvn compile
```

We are mapping the `.m2` repository that maven uses to download dependencies into the `.m2_local` directory so that when we re-run the commands, the container will not download those dependencies again.
Make sure that the tests run with:

```bash
docker run --rm -v $PWD:/home -v $PWD/.m2_local:/root/.m2 -w /home maven:3.3.9 mvn test
```

Then run `PIT` using:

```bash
docker run --rm -v $PWD:/home -v $PWD/.m2_local:/root/.m2 -w /home maven:3.3.9 mvn org.pitest:pitest-maven:mutationCoverage
```

Then head over to the `target/pit-reports` directory and open the `index.html` file to see the report. You will see that line coverage is 100%, but mutation score is 80%, i.e., there is a mutant that is not killed by the existing test suite. Then, improve the test suite, in particular the second test:

```java
package tutorial;

import org.junit.Test;
import static org.junit.Assert.*;

public class CalculatorTest {

    Calculator calc = new Calculator();

    @Test
    public void testAdd() {
        assertEquals(5, calc.add(2, 3));
    }

    @Test
    public void testIsPositive() {
        assertTrue(calc.isPositive(1));
        assertFalse(calc.isPositive(0)); // <-- added assertion to catch the additional edge case
        assertFalse(calc.isPositive(-1));
    }
}
```

and recompile the tests:

```bash
docker run --rm -v $PWD:/home -v $PWD/.m2_local:/root/.m2 -w /home maven:3.3.9 mvn test
```

and then re-run `PIT` using:

```bash
docker run --rm -v $PWD:/home -v $PWD/.m2_local:/root/.m2 -w /home maven:3.3.9 mvn org.pitest:pitest-maven:mutationCoverage
```

Inspect the `index.html` file into `target/pit-reports` to see that mutation coverage is now 100%.

