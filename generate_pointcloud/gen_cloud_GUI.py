import tkinter as tk
import urdfpy
import numpy as np
import rospy
import struct
import time
from tkinter import filedialog
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

class GenereatePointCloud:
    def __init__(self) -> None:
        self.from_extern = False

    def create_ros_node(self):
        rospy.init_node('reachability_pointcloud_publisher', anonymous=True)
        self.pub_points = rospy.Publisher('/reachability_pointcloud', PointCloud2, queue_size=10)
        self.pub_rate = rospy.Rate(1)  # 1 Hz

    def parse_urdf(self):
        # load the urdf file
        self.robot = urdfpy.URDF.load(self.urdf_file_path)
    
    def update_joint_list(self):
        self.joint_list_str = []
        for joint in self.robot.actuated_joints:
            self.joint_list_str.append(joint.name)
        
    def upload_and_active_joint_sel(self):
        # delete all the entries from the menu
        menu = self.wrist_lst_j_menu["menu"]
        menu.delete(0, "end")
        menu = self.arm_lst_j_menu["menu"]
        menu.delete(0, "end")
        menu = self.arm_frt_j_menu["menu"]
        menu.delete(0, "end")

        self.wrist_lst_j.set("")
        self.arm_lst_j.set("")
        self.arm_frt_j.set("")

        # add all the joints to the menus
        for joint in self.joint_list_str:
            menu = self.wrist_lst_j_menu["menu"]
            menu.add_command(label=joint, command=tk._setit(self.wrist_lst_j, joint))
            
            menu = self.arm_lst_j_menu["menu"]
            menu.add_command(label=joint, command=tk._setit(self.arm_lst_j, joint))
            
            menu = self.arm_frt_j_menu["menu"]
            menu.add_command(label=joint, command=tk._setit(self.arm_frt_j, joint))
        
        # activate all widgets
        self.wrist_lst_j_lbl.config(state='active')
        self.wrist_lst_j_menu.config(state='active')
        
        self.arm_lst_j_lbl.config(state='active')
        self.arm_lst_j_menu.config(state='active')

        self.arm_frt_j_lbl.config(state='active')
        self.arm_frt_j_menu.config(state='active')

        # activate the sampling size entry
        self.num_samples_lbl.config(state='active')
        self.num_samples_spinbox.config(state='normal')

        # activate the generate button
        self.gen_cloud_button.config(state='active')

    def select_urdf_file(self):
        # select the desired URDF using the dialog window
        self.urdf_file_path = filedialog.askopenfilename()
        if self.urdf_file_path and self.urdf_file_path[-5:] == ".urdf":
            # write the path in the search bar
            self.urdf_path_entry.delete(0, tk.END)
            self.urdf_path_entry.insert(0, self.urdf_file_path)

            # parse the URDF
            self.parse_urdf()

            # update the joint list
            self.update_joint_list()

            # update all the selectable joint list and activate the widgets
            self.upload_and_active_joint_sel()
        else:
            # write the path in the search bar
            self.urdf_path_entry.delete(0, tk.END)
            self.text_box.config(state='normal')
            self.text_box.insert('end', "Invalid file!\n", 'error')
            self.text_box.yview(tk.END)
            self.text_box.config(state='disabled')

    def generate_point_cloud(self):
        # check if the last wrist joint is selected
        if self.from_extern == False:
            self.wrist_lst_j_name = self.wrist_lst_j.get()
            if self.wrist_lst_j_name==None or self.wrist_lst_j_name=="":
                self.text_box.config(state='normal')
                self.text_box.insert('end', "The last joint of the wrist has not been selected\n",'error')
                self.text_box.yview(tk.END)
                self.text_box.config(state='disabled')
                return

            # check if the last arm joint is selected
            self.arm_lst_j_name = self.arm_lst_j.get()
            if self.arm_lst_j_name==None or self.arm_lst_j_name=="":
                self.text_box.config(state='normal')
                self.text_box.insert('end', "The last joint of the arm has not been selected\n",'error')
                self.text_box.yview(tk.END)
                self.text_box.config(state='disabled')
                return
            
            # check if the first arm joint is selected
            self.arm_frt_j_name = self.arm_frt_j.get()
            if self.arm_frt_j_name==None or self.arm_frt_j_name=="":
                self.text_box.config(state='normal')
                self.text_box.insert('end', "The first joint of the arm has not been selected\n",'error')
                self.text_box.yview(tk.END)
                self.text_box.config(state='disabled')
                return
            
            # check if the numbe rof samples per joint has been entered
            self.num_samples = self.num_samples_spinbox.get()
            if self.num_samples==None or self.num_samples=="":
                self.text_box.config(state='normal')
                self.text_box.insert('end', "The number of samples per joint has not been enetered\n",'error')
                self.text_box.yview(tk.END)
                self.text_box.config(state='disabled')
                return
            self.num_samples = int(self.num_samples)
        
        # retrieve the last wrist joint
        wrist_lst_joint = self.robot.joint_map.get(self.wrist_lst_j_name)

        # let the user select the last arm joint
        arm_lst_joint = self.robot.joint_map.get(self.arm_lst_j_name)
        
        # let the user select the first arm joint
        arm_frt_joint = self.robot.joint_map.get(self.arm_frt_j_name)
        
        # create a dictionary to retrieve the name of the joint from the parent or child link name
        plink_2_joint_name = {}
        clink_2_joint_name = {}
        for joint_name, joint in self.robot.joint_map.items():
            plink_2_joint_name[joint.parent] = joint_name
            clink_2_joint_name[joint.child] = joint_name
        
        # start the timer
        start = time.time()

        # compute the coordinates of the representative point wrt to last arm joint
        curr_joint = wrist_lst_joint
        if curr_joint!=arm_lst_joint:
            rpp_coords = curr_joint.origin[:,3]
        else:
            rpp_coords = np.array([0, 0, 0, 1])

        while(curr_joint!=arm_lst_joint and curr_joint.parent != arm_lst_joint.child):
            prec_joint = self.robot.joint_map.get(clink_2_joint_name[curr_joint.parent])
            T0 = prec_joint.origin
            rpp_coords = np.dot(T0,np.transpose(rpp_coords))
            curr_joint = prec_joint

        points = np.array(rpp_coords)
        points = np.expand_dims(points,1)

        # compute all the possible positions of the representative points
        curr_joint = arm_lst_joint
        while(True):
            T0 = curr_joint.origin
            # create a span of values for the desired joint
            steps = np.linspace(curr_joint.limit.lower, curr_joint.limit.upper, self.num_samples)

            if self.from_extern == False:
                self.text_box.config(state='normal')
                self.text_box.insert('end', "Computing points resulting from rotation of {}... ".format(curr_joint.name))
                self.text_box.yview(tk.END)
                self.text_box.config(state='disabled')

            new_points = []

            # multiply all previous points for the matrix 
            for step in steps:
                T1= np.zeros((4,4))
                T1[3,3] = 1

                if curr_joint.axis[0]==1:
                    T1[:3,:3] = Rotation.from_euler('x',step).as_matrix()
                elif curr_joint.axis[1]==1:
                    T1[:3,:3] = Rotation.from_euler('y',step).as_matrix()
                elif curr_joint.axis[2]==1:
                    T1[:3,:3] = Rotation.from_euler('z',step).as_matrix()

                tmp = np.dot(T1, points)
                tmp = np.dot(T0, tmp)
                new_points.append(tmp)
            if self.from_extern == False:
                self.text_box.config(state='normal')
                self.text_box.insert('end', "done\n")
                self.text_box.yview(tk.END)
                self.text_box.config(state='disabled')

            new_points = np.array(new_points)
            new_points = np.concatenate(new_points, axis=1)
            points = new_points

            if curr_joint == arm_frt_joint:
                break

            if curr_joint.parent in clink_2_joint_name:
                curr_joint = self.robot.joint_map.get(clink_2_joint_name[curr_joint.parent])
            else:
                break


        self.gen_time = time.time()-start

        if self.from_extern == False:
            self.text_box.config(state="normal")
            self.text_box.insert('end', "Point cloud generated in: {}\n".format(self.gen_time))
            self.text_box.yview(tk.END)
            self.text_box.config(state='disabled')

            self.text_box.config(state='normal')
            self.text_box.insert('end', "\nAll arm's joints considered. Total numer of points: {}\n".format(points.shape[1]))
            self.text_box.yview(tk.END)
            self.text_box.config(state='disabled')
        
        self.points = points[:3,:].transpose()
        self.point_cloud_orig_frame = curr_joint.parent

        # activate the publish button
        if self.from_extern == False:
            self.pub_msg.config(state='active')
        
    def create_pointcloud_msg(self):
        points_list = self.points.tolist()
        # points_list.append([np.mean([np.min(points[:,0]), np.max(points[:,0])]),
        #                     np.mean([np.min(points[:,1]), np.max(points[:,1])]),
        #                     np.mean([np.min(points[:,2]), np.max(points[:,2])])])
        # self.points = np.array(points_list)  
          
        # Create PointCloud2 message
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.point_cloud_orig_frame

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]

        self.pointcloud_msg = PointCloud2()
        self.pointcloud_msg.header = header
        self.pointcloud_msg.height = 1
        self.pointcloud_msg.width = self.points.shape[0]
        self.pointcloud_msg.fields = fields
        self.pointcloud_msg.is_bigendian = False
        self.pointcloud_msg.point_step = 12  # 3 * 4 bytes (float32)
        self.pointcloud_msg.row_step = self.pointcloud_msg.point_step * self.pointcloud_msg.width
        self.pointcloud_msg.is_dense = True

        # Pack the point data
        data = []
        for point in self.points:
            data.append(struct.pack('fff', *point))
        self.pointcloud_msg.data = b''.join(data)

    def publish_pointcloud_msg(self):
        self.create_pointcloud_msg()
        self.pub_points.publish(self.pointcloud_msg)
        self.pub_rate.sleep()
        self.text_box.config(state='normal')
        self.text_box.insert('end', "Message published.\n")
        self.text_box.yview(tk.END)
        self.text_box.config(state='disabled')

    def create_GUI(self):
        # create the main window
        self.root_window = tk.Tk(className="Point Cloud generation")
        self.root_window.resizable(False, False)

        # create a frame for the title
        self.title_frame = tk.Frame(self.root_window, height=200, width=1400)
        self.title_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        # create the frame for the search bar
        self.src_urdf_frame = tk.Frame(self.root_window, height=200, width=1400)
        self.src_urdf_frame.grid(row=1, column=0, padx=10, pady=10)

        # create an inner frame
        self.inner_frame = tk.Frame(self.root_window, height=1000, width=1400)
        self.inner_frame.grid(row=2, column=0, padx=10, pady=10)

        # create the frame for the selection of the joint
        self.joint_sel_frame = tk.Frame(self.inner_frame, height=1000, width=680)
        self.joint_sel_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

        # create a title
        self.gui_title = tk.Label(self.title_frame, text="Point Cloud Genereation tool", font=("TkDefaultFont", 18, "bold"))
        self.gui_title.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        # self.description = tk.Label(self.title_frame, text="Tool to generate the point cloud representing the reachability space of the selected manipulator.")
        # self.description.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

        # create a text label and an entry bar
        self.urdf_path_label = tk.Label(self.src_urdf_frame, text="Select a URDF file",height=1)
        self.urdf_path_label.grid(row=0, column=0, padx=10, pady=10)
        self.urdf_path_entry = tk.Entry(self.src_urdf_frame, width=50)
        self.urdf_path_entry.grid(row=0, column=1, padx=10, pady=10)

        # create a button to browse the urdf file
        self.urdf_browse_button = tk.Button(self.src_urdf_frame, text="Browse", height=2, width=10, command=self.select_urdf_file)
        self.urdf_browse_button.grid(row=0, column=2, padx=10, pady=10)

        # wrist last joint
        self.wrist_lst_j = tk.StringVar(self.joint_sel_frame)
        self.wrist_lst_j.set("")
        self.wrist_lst_j_lbl = tk.Label(self.joint_sel_frame, text="Last joint of the wrist:")
        self.wrist_lst_j_menu = tk.OptionMenu(self.joint_sel_frame, self.wrist_lst_j, "")
        self.wrist_lst_j_menu.config(width=20, background="white")
        self.wrist_lst_j_lbl.grid(row=0, column=0, padx=10, pady=20, sticky="e")
        self.wrist_lst_j_menu.grid(row=0, column=1)
        self.wrist_lst_j_lbl.config(state='disabled')
        self.wrist_lst_j_menu.config(state='disabled')

        # arm last joint
        self.arm_lst_j = tk.StringVar(self.joint_sel_frame)
        self.arm_lst_j.set("")
        self.arm_lst_j_lbl = tk.Label(self.joint_sel_frame, text="Last joint of the arm:")
        self.arm_lst_j_menu = tk.OptionMenu(self.joint_sel_frame, self.arm_lst_j, "")
        self.arm_lst_j_menu.config(width=20, background="white")
        self.arm_lst_j_lbl.grid(row=1, column=0, padx=10, pady=20, sticky="e")
        self.arm_lst_j_menu.grid(row=1, column=1)
        self.arm_lst_j_lbl.config(state='disabled')
        self.arm_lst_j_menu.config(state='disabled')

        # arm first joint
        self.arm_frt_j = tk.StringVar(self.joint_sel_frame)
        self.arm_frt_j.set("")
        self.arm_frt_j_lbl = tk.Label(self.joint_sel_frame, text="First joint of the arm:")
        self.arm_frt_j_menu = tk.OptionMenu(self.joint_sel_frame, self.arm_frt_j, "")
        self.arm_frt_j_menu.config(width=20, background="white")
        self.arm_frt_j_lbl.grid(row=2, column=0, padx=10, pady=20, sticky="e")
        self.arm_frt_j_menu.grid(row=2, column=1)
        self.arm_frt_j_lbl.config(state='disabled')
        self.arm_frt_j_menu.config(state='disabled')

        # number of values for each joint
        self.num_samples_lbl = tk.Label(self.joint_sel_frame, text="Samples for each joint:",height=1)
        self.num_samples_lbl.grid(row=3, column=0, padx=10, pady=20, sticky="e")
        self.num_samples_spinbox = tk.Spinbox(self.joint_sel_frame, width=22, from_=1, to=100)

        self.num_samples_spinbox.grid(row=3, column=1, padx=10, pady=10)
        self.num_samples_lbl.config(state='disabled')
        self.num_samples_spinbox.config(state='disabled')

        # button to start the computation
        self.gen_cloud_button_frm = tk.Frame(self.joint_sel_frame, height=50, width=50)
        self.gen_cloud_button_frm.grid(row=4, column=0, columnspan=2, sticky="s")

        self.gen_cloud_button = tk.Button(self.gen_cloud_button_frm, text="Generate", height=2, width=10, command=self.generate_point_cloud)
        self.gen_cloud_button.grid(row=0, column=0, padx=20, pady=20)
        self.gen_cloud_button.config(state='disabled')

        # button to publish the desired message
        self.pub_msg = tk.Button(self.gen_cloud_button_frm, text="Publish", height=2, width=10, command=self.publish_pointcloud_msg)
        self.pub_msg.grid(row=0, column=1, padx=20, pady=20)
        self.pub_msg.config(state='disabled')

        # create a text box where to print debug text
        self.text_box_frame = tk.Frame(self.root_window, height=200, width=1400)
        self.text_box_frame.grid()
        self.text_box = tk.Text(self.text_box_frame, height=10)
        self.text_box.grid(row=0, column=0, pady=20)
        self.text_box.tag_config('error', foreground="red")

        self.text_box_scrlbr = tk.Scrollbar(self.text_box_frame, command=self.text_box.yview, width=20)
        self.text_box_scrlbr.grid(row=0, column=1, sticky="nsew", pady=20)
        self.text_box.config(yscrollcommand=self.text_box_scrlbr.set)
        self.text_box.yview(tk.END)
        self.text_box.config(state='disabled')

        # start the main loop
        self.root_window.mainloop()

if __name__=="__main__":
    gen_cloud = GenereatePointCloud()
    gen_cloud.create_ros_node()
    gen_cloud.create_GUI()
