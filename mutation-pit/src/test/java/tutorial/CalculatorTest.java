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
        assertFalse(calc.isPositive(-1));
    }
}