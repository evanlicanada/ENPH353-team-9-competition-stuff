#!/bin/bash

echo "Starting terminals"
pkill -9 gzserver || true
pkill -9 gzclient || true
pkill -9 ros || true

sleep 2

# --- Command 1: Run the sim ---
xfce4-terminal --title="gazebo_terminal" --command="bash -c 'source ~/ros_ws/devel/setup.bash; cd ~/ros_ws/src/2025_competition/enph353/enph353_utils/scripts; ./run_sim.sh -vpgw; exec bash'" &
sleep 2
# --- Command 2: Run the score tracker ---
xfce4-terminal --title="score_tracker" --command="bash -c 'source ~/ros_ws/devel/setup.bash; cd ~/ros_ws/src/2025_competition/enph353/enph353_utils/scripts; ./score_tracker.py; exec bash'" &



echo "Press enter to start robot..."
# --- Running the robot and camera view ---
read
# --- Command 3: Camera view ---
xfce4-terminal --title="camera_view" --command="bash -c 'source ~/ros_ws/devel/setup.bash; sleep 2; rosrun rqt_image_view rqt_image_view; exec bash'" &
source ~/ros_ws/devel/setup.bash
roslaunch robot_ctrl_pkg robot.launch  

echo "All three services have been launched in the background."
echo "You can close this initial script window now, but the new windows will remain open."