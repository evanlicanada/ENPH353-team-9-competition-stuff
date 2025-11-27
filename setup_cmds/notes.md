# Just some notes to make it so I have to remember less

## Lanching the environment
So this is a pretty annoying task requiring at least three terminals (and a 4th for camera view), so instead just open a terminal in the parent folder of this file, then run `./launch_package.sh`
- Note: make sure it has executable permissions
- This will:
1. (hopefully) clean up the old ros and gazebo stuff, 
2. then launch the gazebo environment and score tracker in seperate terminals
3. The main terminal will wait for you to press enter
4. once pressed, it will launch the package and open the camera view

Note: I might want to change it so that the package is launched right away, but the robot waits for enter... or maybe it doesn't matter idk

Oh one more thing, all the new terminal windows should be closed before running the script again, it should still work but it'll really clutter up the desktop