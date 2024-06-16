# Find animals in your security camera footage using Megadetector

I live in an area of the United States where I have a lot of random wildlife
wandering through my backyard. At the end of 2022, I put up security cameras so
I could spy on those little cuties. I very quickly realized, however, that
there's a whole lot of nothing that happens in between the times when the
fluffy little guys come sauntering through. I started thinking about it and, lo
and behold, someone out there on the internet has solved this problem for me.
The [MegaDetector](https://github.com/agentmorris/MegaDetector) project has a
pretty nifty AI model for finding animals in images. They've also provided some
Python code to demonstrate how to use it. And that, in a nutshell, is what this
repo is: me leveraging what I know about running distributed computer programs
with a nifty AI model that I can use to find animals in my backyard video
cameras.

As of the summer of 2024, I'm still trying to figure out how to speed it up
without spending thousands on GPU instances on AWS (or a suped up desktop or
something) and, honestly, I think I've reached the limits of what my laptop
equipped with an Nvidia RTX 2070 can do. To process around 90 minutes of video
is currently taking me about 4.5 hours on my laptop. **However** ... remember
how I said I started with what I know about running distributed computer
programs? Well, I put together a Dockerfile and docker-compose.yaml that one
_could_ use to run this across several GPU equipped EC2 instances or something.
Note that this probably requires a trust fund or some other means by which you
can finance this without impacting your ability to pay rent. I processed
exactly _one_ approximately 90 minute long video on a XXXXXXX instance and my
laptop and it took XXXXXX. So, I guess you can use that information however you
want to. I'll get into the nitty gritty of how to set this project up to run
that way further down in the README but first let's talk about what it takes to
run it in the first place.

### Prerequisites

At the very least, you'll need a Nvidia GPU equipped computer to run this on.
I've never attempted to run it on anything other than Ubuntu 22.04 with version
12.5 of the Nvidia CUDA Toolkit but I think it'll probably work with older
versions as well. Getting that setup can be a little weird but most concise
instructions I've found are
[here](https://developer.nvidia.com/cuda-downloads). I make no claims to know
very much about that process but if you get stuck, maybe we can put our heads
together and figure it out (just open an issue).

One other thing that has been a bit of a struggle for me with this is having
enough space on a fast enough disk. Firstly, extracting all of the frames from
hours of video takes up a whole bunch of disk space. I tried to make it so the
JPEG compression that the project uses is good enough to retain quality but
also make the files a little smaller. That said, a video of approximately 90
minutes has over 100,000 images in it (and that's at a pretty low frame rate).
In my experience, this can consume around 180GB of disk space. 

The other thing that ends up requiring a lot of disk space is the docker
images. Besides space, the disk needs to be relatively fast. So, if you're
thinking "I've got a big 'ol USB backup drive I can use for this", just be
prepared to wait because having fast disk ends up making a real difference.
Further down when I talk about how to distribute this across several systems,
I'll talk about how to use something like AWS S3 for a storage backend. This is
_fine_ I guess but, again, using a local, fast disk really makes it better.

To use the Docker setup, you'll also need to install the [Nvidia Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
FWIW, that's probably going to be the vastly simpler way of using this (and, to
be honest, probably the better way, too, since it'll be easier to scale and run
across several systems that way, too). You also won't need to worry about
getting the CUDA Toolkit setup on your host since the container handles all of
that for you and just leverages whatever GPU you have on your host. 

So, to summarize for the people who are just skimming, the prerequisites are:

* Nvidia GPU
* Debian-ish Linux distro (tested on Ubuntu 22.04)
* [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (tested with 12.5 but older versions will probably work)
* A lot of disk space on a fast disk (optional but, like, definitely worth it)
* For Docker: [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### How this project is stitched together

The short version is that this project uses Django + Celery to run a coule
different "chunks" of the processing pipeline in a more or less distributed
way. Could it be more distributed? Probably. But I'm not really willing to give
AWS that kind of money (yet). You'll see these stages reflected in the [celery
tasks](processor/tasks.py) and in the names of the services in the
[`docker-compose.yaml`](docker-compose.yaml) file. These stages look like this:

**Extract frames from the video** As I'm sure you've guessed, this iterates
through the video and saves out each frame as a separate image and makes a row
for each frame in the DB. This is really where the storage backend you're using
will come into play since, as I mentioned above, an approximately 90 minute
video with a framerate of 20 fps will use up about 180GB of disk. If you're
using S3, you'll need to consider the data transfer costs, too (unless
you're doing _all_ your processing in AWS). Right now, this is just handled
by a single worker since the `decord` library does a pretty amazing job of
quickly extracting the frames far faster than the detection workers can run
detections on them. If you're getting to the point where you have enough
detection workers that this process can't keep up, open an issue and we can
talk about how to parallelize this process.  

**Detect whether or not there are interesting things in the images** This is
the meat and potatoes of the process. It uses the MegaDetector v5a model to
figure out if there are likely to be animals, people, or vehicles in each frame
of the video and, if it finds things, it saves its findings to a DB table. What
do its findings look like? Here's what the DB table looks like: 

```
  id  | category | confidence |  x_coord  | y_coord | box_width | box_height | frame_id 
------+----------+------------+-----------+---------+-----------+------------+----------
  319 | 2        |      0.699 |   0.08125 |  0.3629 |      0.05 |     0.3407 |     1331
  366 | 2        |      0.793 |   0.07812 |  0.3638 |   0.05312 |     0.3407 |     1332
  314 | 2        |      0.793 |   0.07812 |  0.3638 |   0.05312 |     0.3407 |     1333
  348 | 2        |      0.736 |   0.07604 |  0.3638 |   0.05572 |     0.3388 |     1334

... etc ...
```

The `category` is what kind of thing (1 = animal, 2 = person, 3 = vehicle) is
represented by the detection. The `confidence` is how confident the model was
that it is, in fact, the thing that it thinks it is. The coordinates represent
the upper left hand corner of where the detection was in the image and you can
use the `box_width` and `box_height` to figure out how big the box is. These
are ultimately something that can be used by PIL to actually go a draw a box on
the image, if that's your thing. I'll be working on a piece of the pipeline
here soon to actually take care of that and output a video but I haven't gotten
there yet. 

### How do I make it go?

If you're using the Docker setup, you should get an `admin` container for free.
By default, it's configured to make a user for you if you set
`DJANGO_SUPERUSER_USERNAME`, `DJANGO_SUPERUSER_PASSWORD` and
`DJANGO_SUPERUSER_EMAIL` in your `.env` file. Then you should be able to login
using those creds by navigating to `http://127.0.0.1:8000/admin` in your web
browser. If you're familiar with the Django Admin, this should look familiar.
Once you're logged in, you can click through to `Video` and then `Add Video` in
the upper right hand corner. From there, you should be prompted for a file to
upload. Once the video is uploaded, it should start processing. 

One thing to note: If you're using more than one machine to process things,
you'll need to use the S3 backend for storage by setting the `AWS` env vars
shown in the example `.env` file. You'll also need to make sure that whatever
creds you are using have the ability to add things to S3 and whatever machines
are doing the processing have access to the bucket(s) you are using to store
the frames and videos. 

If you're running this thing _entirely_ on AWS which makes the admin container
a little hard to access, open an issue and I can help you get it set up such
that the admin is exposed in a way that you can access it. If you're familiar
with how to deploy a Django app in production using docker compose, then you
probably don't need my help.

### Docker build process
This project relies upon [`decord`](https://github.com/dmlc/decord) to quickly
extract frames from your video files. In order to enable GPU acceleration for
that library, you need to install it from source. The Dockerfile included here
will take care of that for you however, you need to download the Nvidia Video
Codec SDK and stick it into the decord folder before you build the docker
image. Why can't the Dockerfile just take care of that for you? Because Nvidia
wants your email address. Anyways, it's pretty simple:

* Recursively clone the decord repo:
```
git clone --recursive https://github.com/dmlc/decord
```
* Go to the [Nvidia Video Codec SDK download
  page](https://developer.nvidia.com/nvidia-video-codec-sdk/download) and
  download the "Video Codec for application developers". It will involve
  registering with Nvidia (Boo!)
* Copy the zip file you end up with to the directory where you cloned the decord repo
* Unzip it
* Build the docker image for this project:
```
docker build -t video-processor:latest .
```

The build will probably take around 5 minutes and use around 10GB of disk.

### Running across several machines


