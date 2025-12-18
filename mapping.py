import cv2
import numpy as np
import matplotlib.pyplot as plt





fig, ax, line, points_scatter = None, None, None, None


# Matplotlib visualization code
def init_3d_plot():
    global fig,ax,line,points_scatter

    plt.ion()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111,projection='3d')

    #Initialize the camera trajectory line plot
    # Set the initial data to empty lists
    line_list = ax.plot([],[],[],'b-',label='Camera Trajectory')
    line = line_list[0]

    # Initialize a points 3d map scatter plot
    points_scatter = ax.scatter([],[],[],marker='.',c='r',s=1)

    ax.set_xlabel('X (Left / right)')
    ax.set_ylabel('Y (Depth)') # Note: Y is usually up/down in camera space, but let's use it for Z-depth here for map clarity
    ax.set_zlabel('Z (Up/Down)')

    # Set initial limits
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-2,2])

    ax.legend()
    plt.title('SLAM Map and Camera Trajectory')
    plt.show(block=False)



def update_3d_plot(slam_map_instance):
    global ax,fig,points_scatter,line

    if not fig:
        return


    # Update trajectory
    # slam_map.trajectory is a list of [X,Y,Z] vectors
    if slam_map_instance.trajectory:
        path = np.array(slam_map_instance.trajectory)

        x_traj = path[:,0]
        y_traj = path[:,1]
        z_traj = path[:,2]

        line.set_data(x_traj,y_traj)
        line.set_3d_properties(z_traj)


        # -- Dynamic plot limits ---

        min_coords = path.min(axis=0)
        max_coords = path.max(axis=0)


        buffer = 1
        ax.set_xlim([min_coords[0] - buffer, max_coords[0] + buffer])
        ax.set_ylim([min_coords[1] - buffer,max_coords[1] + buffer])
        ax.set_zlim([min_coords[2] - buffer, max_coords[2] + buffer])

    # --- Update cloud point ----
    if slam_map_instance.points_3d is not None and len(slam_map_instance.points_3d) > 0:
        all_points = slam_map_instance.points_3d.T # Transpose to get 3 rows (X,Y,Z)

        points_scatter._offsets3d = (all_points[0],all_points[1],all_points[2])

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)




def show_final_map(slam_map):
    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(111, projection='3d')

    if len(slam_map.trajectory) > 0:
        path = np.array(slam_map.trajectory)


        mask = np.linalg.norm(path, axis=1) < 5
        ax.plot(path[mask, 0], path[mask, 1], path[mask, 2], 'b-', linewidth=2)


    if len(slam_map.points_3d) > 0:
        pts = np.array(slam_map.points_3d)
        mask = np.linalg.norm(pts, axis=1) < 5
        ax.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2], c='r', s=1, alpha=0.5)


    # Set fixed limits so we don't zoom out to space
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_box_aspect([1,1,1])
    plt.show()




# CONSTANTS (adjust depending on your setup)
FOCAL_LENGTH = 1100.0
CX = 640.0
CY = 360.0

# Camera's properties for 2D-3D conversion
K = np.array([
    [FOCAL_LENGTH,0,CX],
    [0,FOCAL_LENGTH,CY],
    [0,0,1]
],dtype=np.float32)




class RoomMapper:
    def __init__(self):
        self.points_3d = [] # Store 3D points from the video
        self.trajectory = [] # Store the camera trajectory
        self.descriptors_3d = []
        self.orb = cv2.ORB_create(3000) # To look for features
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True) # Feature matcher


    def extract_features(self,frame):
        features = self.orb.detectAndCompute(frame,None)
        return features



    def process_first_frame(self,frame1,frame2):
        # Get features from both frames
        kp1,des1 = self.extract_features(frame1)
        kp2,des2 = self.extract_features(frame2)

        # Match them
        matches = self.bf.match(des1,des2)
        matches = sorted(matches, key=lambda x: x.distance)


        # Estimate Motion (Rotation and Translation)
        R, t, mask, pt1, pt2 = self.estimate_motion(kp1, kp2, matches)


        # Triangulation : Convert 2D movement into 3D Points
        # Proj1 is the origin (the point where camera started from)
        proj1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        proj2 = np.hstack((R,t))

        # projects pixels into 3D using camera matrix K
        pts1_norm = cv2.undistortPoints(pt1.reshape(-1,1,2),K,None)
        pts2_norm = cv2.undistortPoints(pt2.reshape(-1,1,2),K,None)

        # The dots
        points_4d = cv2.triangulatePoints(proj1,proj2,pts1_norm,pts2_norm) # This returns 4D matrix
        points_3d = (points_4d[:3] / points_4d[3]).T # Normalize from 4D to 3D

        # Descriptors for the triangulated points


        points_3d = np.array(points_3d)

        mask =  mask.ravel()
        self.points_3d = points_3d[mask == 1]

        good_matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]
        self.descriptors_3d = np.array([des2[m.trainIdx] for m in good_matches],dtype=np.uint8)



        return points_3d,R,t


    # Performs the essential matrix calculation for us (gives us the R and t)

    # Go find every 'bridge' we built between these two frames, get the $(x, y)$ location of where
    # that bridge starts
    # in Frame 1 and where it ends in Frame 2, and put them into two matching lists."

    def estimate_motion(self,kp1,kp2,matches):
       # Converts matches into coordinates
      pt1 = np.float32([kp1[m.queryIdx].pt for m in matches])
      pt2 = np.float32([kp2[m.trainIdx].pt for m in matches])

      E,mask = cv2.findEssentialMat(pt1,pt2)

      _,R,t,_ = cv2.recoverPose(E,pt1,pt2,K)

      return R,t,mask,pt1,pt2


    def track_camera(self,kp,des):
        "Uses Solve PnP to find camera position based on existing 3D Map"

        if des is None or len(des) < 20 or len(self.descriptors_3d) < 20:
            return None,None

        map_des = np.array(self.descriptors_3d,dtype=np.uint8)
        matches = self.bf.match(map_des,des)

        # Need atleast 4 maps, we'll do 20 for stability
        if len(matches) < 20:
            return None,None

        # Align 3D points with their 2D locations in the curretn frame
        obj_pts = np.float32([self.points_3d[m.queryIdx] for m in matches])
        img_pts = np.float32([kp[m.trainIdx].pt for m in matches])

        # Solve for Pose (Rotation and image)
        _,rvec,tvec,inliers = cv2.solvePnPRansac(obj_pts,img_pts,K,None)

        # Convert rotation vector into matrix
        R,_ = cv2.Rodrigues(rvec)

        return R,tvec


    def grow_map(self,frame1,frame2,R_curr,t_curr):
        if frame1 is None : return

        kp1,des1 = self.extract_features(frame1)
        kp2,des2 = self.extract_features(frame2)

        matches = self.bf.match(des1,des2)

        # only keep the best new features
        matches = sorted(matches,key=lambda x : x.distance)[:100]

        # Current movement matrices
        proj1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        proj2 = np.hstack((R_curr,t_curr))

        if np.any(np.isinf(proj2)) or np.any(np.isnan(proj2)):
            return

        pt1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pt2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        pts1_norm = cv2.undistortPoints(pt1.reshape(-1,1,2),K,None)
        pts2_norm = cv2.undistortPoints(pt2.reshape(-1,1,2),K,None)

        # Create new 3D dots
        points_4d = cv2.triangulatePoints(proj1,proj2,pts1_norm,pts2_norm)
        w = points_4d[3]
        w[np.abs(w) < 1e-5] = 1e-5
        new_pts_3d = (points_4d[:3] / w).T

        # ADD TO OUR MASTER LIST
        self.points_3d = np.vstack((self.points_3d,new_pts_3d))

        new_des = np.array([des2[m.trainIdx] for m in matches],dtype=np.uint8)
        self.descriptors_3d = np.vstack((self.descriptors_3d,new_des))





if __name__ == "__main__":
    video_path = 'room_walk.mp4'
    cap = cv2.VideoCapture(video_path)

    mapper = RoomMapper()

    frame_count = 0
    init_frame = None
    is_initialized = None

    init_3d_plot()

    cur_R = np.eye(3)
    cur_t = np.zeros((3,1))

    prev_gray_key = None

    while cap.isOpened():
        ret,frame = cap.read()

        if not ret : break

        # Convert the frame to gray
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        kp,des = mapper.extract_features(gray)

        # Get the very first frame
        if frame_count == 0:
            init_frame = (gray,kp,des)

        # Initalize the 3D Map (frame 20 for enough movement)
        if frame_count == 20 and is_initialized is None:
            points_3d,R,t = mapper.process_first_frame(init_frame[0],gray)

            mapper.points_3d = points_3d

            prev_gray_key = gray

            is_initialized = True
            print('SLAM Initialized')

        elif is_initialized:
            R_curr,t_curr = mapper.track_camera(kp,des)

            if t_curr is not None:
                R_w = R_curr.T
                t_w = -R_w @ t_curr
                new_pos = t_w.flatten()

                if len(mapper.trajectory) > 0:
                    prev_pos = np.array(mapper.trajectory[-1])
                    movement = np.linalg.norm(new_pos - prev_pos)
                    if movement > 0.5:
                        t_curr = None


                if t_curr is not None:
                  mapper.trajectory.append(t_w.flatten())

                  if frame_count % 10 == 0:
                    mapper.grow_map(prev_gray_key,gray,R_curr,t_curr)
                    prev_gray_key = gray.copy()

                  cv2.putText(frame,f'Pos : {t_curr.flatten()}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)


        # Visualize features
        for p in kp:
            u,v = map(int,p.pt)
            cv2.circle(frame,(u,v),2,(0,255,0),-1)


        cv2.imshow('Mapping Room....',frame)

        # Update for next frame
        prev_gray = gray.copy()
        prev_kp = kp
        prev_des = des
        frame_count += 1



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print('Mapping completed. press q to check the 3D plot')
    plt.ioff()
    show_final_map(mapper)








