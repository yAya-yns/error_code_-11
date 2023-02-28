Please refer to `lab2_tasks_v3.pdf` for project objectives.

## Starting in docker:

```cd docker && bash start_docker```


## Starting: 
Run in seperate terminal (using docker):

```roslaunch rob521_lab2 willowgarage_world.launch```

```roslaunch rob521_lab2 map_view.launch```

## Run unit test:

```
cd ~/catkin_ws/src/lab2/nodes
python3 unit_tests/ut_xxxx.py
```

Potential Error:
```
Traceback (most recent call last):
  File "unit_tests/ut_is_collide.py", line 1, in <module>
    from l2_planning import *
ModuleNotFoundError: No module named 'l2_planning'
```

Solution: 
```
export PYTHONPATH="${PYTHONPATH}:/path/to/folder/"
```


