# People_Counter_SSD
a “people counter” with OpenCV and Python. Using OpenCV, we’ll count the number of people who are heading “in” or “out” of a department store in real-time.  using SSD model to detect the person label class and then track using an algolirthm is cenroids_tracker

Combining both object detection and object tracking

Highly accurate object trackers will combine the concept of object detection and object tracking into a single algorithm, typically divided into two phases:

    Phase 1 — Detecting: During the detection phase we are running our computationally more expensive object tracker to (1) detect if new objects have entered our view, and (2) see if we can find objects that were “lost” during the tracking phase. For each detected object we create or update an object tracker with the new bounding box coordinates. Since our object detector is more computationally expensive we only run this phase once every N frames.
    Phase 2 — Tracking: When we are not in the “detecting” phase we are in the “tracking” phase. For each of our detected objects, we create an object tracker to track the object as it moves around the frame. Our object tracker should be faster and more efficient than the object detector. We’ll continue tracking until we’ve reached the N-th frame and then re-run our object detector. The entire process then repeats.

The benefit of this hybrid approach is that we can apply highly accurate object detection methods without as much of the computational burden. We will be implementing such a tracking system to build our people counter.

To implement our people counter we’ll be using both OpenCV and dlib. We’ll use OpenCV for standard computer vision/image processing functions, along with the deep learning object detector for people counting.

We’ll then use dlib for its implementation of correlation filters. We could use OpenCV here as well; however, the dlib object tracking implementation was a bit easier to work with for this project.

I’ll be including a deep dive into dlib’s object tracking algorithm in next week’s post.

Along with dlib’s object tracking implementation, we’ll also be using my implementation of centroid tracking from a few weeks ago. Reviewing the entire centroid tracking algorithm is outside the scope of this blog post, but I’ve included a brief overview below.

At Step #1 we accept a set of bounding boxes and compute their corresponding centroids (i.e., the center of the bounding boxes):
Figure 2: To build a simple object tracking via centroids script with Python, the first step is to accept bounding box coordinates and use them to compute centroids.

The bounding boxes themselves can be provided by either:

    An object detector (such as HOG + Linear SVM, Faster R- CNN, SSDs, etc.)
    Or an object tracker (such as correlation filters)

In the above image you can see that we have two objects to track in this initial iteration of the algorithm.

During Step #2 we compute the Euclidean distance between any new centroids (yellow) and existing centroids (purple):
Figure 3: Three objects are present in this image. We need to compute the Euclidean distance between each pair of original centroids (red) and new centroids (green).

The centroid tracking algorithm makes the assumption that pairs of centroids with minimum Euclidean distance between them must be the same object ID.

In the example image above we have two existing centroids (purple) and three new centroids (yellow), implying that a new object has been detected (since there is one more new centroid vs. old centroid).

The arrows then represent computing the Euclidean distances between all purple centroids and all yellow centroids.

Once we have the Euclidean distances we attempt to associate object IDs in Step #3:
Figure 4: Our simple centroid object tracking method has associated objects with minimized object distances. What do we do about the object in the bottom-left though?

In Figure 4 you can see that our centroid tracker has chosen to associate centroids that minimize their respective Euclidean distances.

But what about the point in the bottom-left?

It didn’t get associated with anything — what do we do?

To answer that question we need to perform Step #4, registering new objects:
Figure 5: In our object tracking example, we have a new object that wasn’t matched with an existing object, so it is registered as object ID #3.

Registering simply means that we are adding the new object to our list of tracked objects by:

    Assigning it a new object ID
    Storing the centroid of the bounding box coordinates for the new object

In the event that an object has been lost or has left the field of view, we can simply deregister the object (Step #5).

Exactly how you handle when an object is “lost” or is “no longer visible” really depends on your exact application, but for our people counter, we will deregister people IDs when they cannot be matched to any existing person objects for 40 consecutive frames.

Again, this is only a brief overview of the centroid tracking algorithm.

Note: For a more detailed review, including an explanation of the source code used to implement centroid tracking, be sure to refer to this post.
Creating a “trackable object”

In order to track and count an object in a video stream, we need an easy way to store information regarding the object itself, including:

    It’s object ID
    It’s previous centroids (so we can easily to compute the direction the object is moving)
    Whether or not the object has already been counted
