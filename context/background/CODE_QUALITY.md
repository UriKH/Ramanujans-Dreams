
# File description

> **Doc type:** Policy / coding standards. Stable rules for how code should
> be written and documented. Update only when the standards themselves
> change.

# Documentation
## Function 
Each function is documented in the following format:
```
"""
General function description
:param <1st param>: param description
:param <2nd param>: param description
...
:raise <Exception 1>: Description on when the excpetion is being thrown.
:raise <Exception 2>: Description on when the excpetion is being thrown.
... 
:return: What the function returns if any.
"""
```

## Class
Each class should have a few lines of short and concice description explaining what the class represents and what is the context.

## Code
- Document nonetrivial code sections and lines.
- Divide big and complex functions to smaller ones to keep things clear and readable.
- When relevant consider using optimizations such as prallelization and vectorization (via tools like numpy and numba). Ask user before doing something nonetrivial optimizations.

# Testing
Code should be thoroughly tested!

- Classes: Each class should have a test class to match it.
- Methods: Each method (class method, utility etc.) shuold be tested with basic as well as advanced edge case tests. 
- All tests shuold have readable descriptive names and be well documented and explain the rational behind the test case.
- When finding missing tests report to user and explain which tests are missing and why in [MISSING_TESTS.md](context/background/TESTS.md).

# Design
The design should be clean and utilize OOP design patterns when possible to simplify the code and allow expandability. 

# Clarity
Code should be clean and readable - variables and function names should be meaningful.

# Expection managment
Raise exceptions with descriptive information to the user . 

For example in the following code is not good:
 If `db.identify()` failes we log a warning and return `None`. We mist not try to mitigate the error by returning a default value!
 Even worse is to return a magic value as a default which supresses the error.

```python
try:
    res = db.identify([low_res_constant] + walk_values[1:])
except Exception as e:
    Logger(f'Error while identifing constnat. LIReC failed with: "{e}"', Logger.Levels.warning).log()
    return None
```

The expected solution should be as follows:
1. Create a unique error class: `LIReCError` (since `db.identify` is part of LIReC library).
    ```python
    class LIReCError: # (<Exception or something else>):
        def __init__(*args, log_level=None, **kwargs):
            if log_level: # Add here a check if a message was passed
                Logger(<here insert the message passed to the constructor>, log_level).log()
            super().__init__(*args, **kwargs)
    ```
2. In the current calling method:
    ```python
    def do_something():
        """
        # ...
        :raise LIReCError: If fails to identify the constant.
        # ...
        """
    # ...

    try:
        return db.identify([low_res_constant] + walk_values[1:])
    except Exception as e:
        raise LIReCError(f'Error while identifing constant. LIReC failed with: "{e}"', Logger.Levels.warning)
    ```
3. In the parent calling function:
    ```python

    try:
        res = do_something()
    except LIReCError as e:
        pass # Choose how to act: pass, raise an error, etc.
    ```