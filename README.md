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
exactly _one_ approximately 90 minute long video on a `g5.xlarge` instance and
my laptop and it took XXXXXX. So, I guess you can use that information however
you want to. I'll get into the nitty gritty of how to set this project up to
run that way further down in the README but first let's talk about what it
takes to run it in the first place.

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

The other thing that ends up requiring a lot of disk space are the docker
images. The image that the Dockerfile in this repo builds is over 10GB so when
you're messing around with it and building different versions, it can add up.

Besides space, the disk needs to be relatively fast. So, if you're thinking
"I've got a big 'ol USB backup drive I can use for this", just be prepared to
wait because having fast disk ends up making a real difference. Further down
when I talk about how to distribute this across several systems and the setup
seamlessly uses AWS S3 for a storage backend. This is _fine_ I guess but,
again, using a local, fast disk really makes it better.

To use the Docker setup, you'll also need to install the [Nvidia Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
FWIW, that's probably going to be the vastly simpler way of using this (and, to
be honest, probably the better way, since it'll be easier to scale and run
across several systems that way, too). You also won't need to worry about
getting the CUDA Toolkit setup on your host since the container handles all of
that for you and just leverages whatever GPU you have on your host. 

So, to summarize for the people who are just skimming, the prerequisites are:

* Nvidia GPU
* Debian-ish Linux distro (tested on Ubuntu 22.04)
* [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (tested with 12.5 but older versions will probably work)
* A lot of disk space on a fast disk (optional but, like, definitely worth it)
* For Docker: [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### How do I make it go?

The simplest way to run this is just on your local machine using the
`docker-compose-local.yaml` file with the `.env.local` file to populate the env
vars. To run it in a more distributed way, see the section on running it in a
more distributed way below. To get a basic running version up, here's the
tl;dr:

* Build the docker image (see "Docker build process" below)
* Make a copy of the example local env file and make changes as needed
  (probably the only thing you'll want to think about changing is the
  `STORAGE_MOUNT` but, it should just work as is)
```
cp .env.local.example .env.local
```
* Run the docker compose file using the copy of the env file you just made:

```
docker compose -f docker-compose-local.yaml --env-file .env.local
```

That's basically it. You should get an `admin` container for free.  By default,
it's configured to make a user for you if you set `DJANGO_SUPERUSER_USERNAME`,
`DJANGO_SUPERUSER_PASSWORD` and `DJANGO_SUPERUSER_EMAIL` in your `.env.local`
file. Then you should be able to login using those creds by navigating to
`http://127.0.0.1:8000/admin` in your web browser. If you're familiar with the
Django Admin, this should look familiar.  Once you're logged in, you can click
through to `Video` and then `Add Video` in the upper right hand corner. From
there, you should be prompted for a file to upload. Once the video is uploaded,
it should start processing. 

### Running in a more distributed way 

The example I've included for running this in a distributed way is running some
workers on an AWS EC2 instance with a GPU and some workers on a local machine.
If you'd like to run this _entirely_ on AWS or another cloud provider, one
thing you'll need to do is make the `admin` container slightly less dumb. Right
now it's just using the Django development server and isn't behind a web
server, isn't using SSL, etc, etc. I'd really, really recommend _not_ just
running that as is anywhere but your local machine. I've been deploying Django
in production environments since 2009 and have done this in Docker a few times
as well so if you get stuck attempting to Google for it, open an issue and I'll
give you some pointers.

At any rate, the tl;dr to get this running on AWS is:

* **Ask AWS to let you spin up GPU instances** If you don't already have
  permission, this is something that you have to submit a support ticket to get
  turned on. They seem to get back to you pretty quickly but, it's a step
  nonetheless. If you're not familiar with how they do these things, you're
  asking for the ability to run a certain number of vCPUs of a given instance
  type. I was able to get 8 vCPUs "granted" to me for "G" type instances (which
  are the cheapest GPU instances as of early 2024).
* **Spin up your GPU instance and install things** I've included instructions
  for getting things setup on an Ubuntu 22.04 machine above (see
  "Prerequisites") and it should work in more or less the same way if
  you're using Ubuntu 22.04 for your new instance. One nice thing that is
  included with the "G" type instances is a 250GB instance store which I
  started using for my docker setup so that I didn't have to pay for a
  massive EBS volume. If you want to do something similar, you can format
  and mount that device and then add a `/etc/docker/daemon.json` file that
  tells your docker setup where to cache the images. I'll let you go ahead
  and Google that so this tl;dr doesn't get too long.
* **Build the docker image** You can either do this locally and push the image
  to AWS's Container Registry (aka ECR; that's what I was doing) or just build
  the image on your new instance. Either way, you can follow the Docker build
  process below. Before you push to ECR, you'll need to get the login creds and
  configure docker to use them. Here's a nifty one-liner for that:
  ```
  aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin <your-ecr-hostname>
  ```
* **Make a DB instance** You should be able to use the smallest instance type
  available to run PostgreSQL or, if you're fancy (and really like giving AWS
  money) you can use RDS or Aurora. You should be able to just spin up a Linux
  instance of your choosing and install PostgreSQL on it. You'll want to edit
  your `pg_hba.conf` file to allow your GPU instance(s) and your local machine
  to connect to it. Also make sure it uses the same security group as the GPU
  instance(s) you spun up. That'll make the next step easier.
* **Setup your Security Group** You'll want to add a couple rules. One that
  allows instances that are associated with that security group to connect to
  one another on port 5432, and then one that allows your local machine to
  connect to it on port 5432. I suppose you could make a couple security groups
  and make that a little cleaner but, ya know, let's just get to the good part,
  shall we?
* **Copy the example .env file** Similar to running this thing locally, you'll
  need to make a copy of `.env.aws` and make changes to it as needed. At the
  very least you'll need to change:
    - `AWS_ACCESS_KEY_ID`
    - `AWS_SECRET_ACCESS_KEY`
    - `AWS_REGION`
    - `FRAMES_BUCKET`
    - `VIDEOS_BUCKET`
    - Probably most of the `DB_*` vars based on how you setup your DB instance
    - `WORKERS_PER_HOST` (currently does nothing but it might sime day)
    - `HOST_COUNT` (ditto)
* **Run it!** You should be able to run the AWS version of the docker compose
  file along with the AWS version of the .env file on the GPU instance(s) as
  well as your local machine like so:
```
docker compose -f docker-compose-aws.yaml --env-file .env.aws
```
* **Process a file** This is the same as on a local setup. Just navigate to
  `http://127.0.0.1:8000/admin` on your local machine, login and then click
  through the UI to upload a new Video. 

That probably glosses over some details but if you're comfortable with AWS and
OK at Googling, you should be able to get things going. If not, open an issue
and I'll see if I can help you out.

### OK, I've processed a video, now what?

The main output of this is a DB table that records where in an image the
detector found something and what kind of a thing it is. What does it look
like? Here's what the DB table looks like: 

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

The build will probably take around 5-10 minutes and use around 10GB of disk.

### How this project is stitched together

The short version is that this project uses Django + Celery to run a couple
different "chunks" of the processing pipeline in a more or less distributed
way. Could it be more distributed? Probably. But I'm not really willing to give
AWS that kind of money (yet). You'll see these stages reflected in the [celery
tasks](processor/tasks.py) and in the names of the services in the
`docker-compose` files. These stages look like this:

**Analyze the video** Really all this does is takes the inputs you give for
`WORKERS_PER_HOST` and `HOST_COUNT` and figures out how to distribute the work
of extracting frames from the video across the workers that you have available.
I suppose there could be some additional steps in there eventually but that's
enough for now. **This step is currently turned off since it really seemed like
it wasn't helping (in fact it probably made things worst)** 

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
of the video and, if it finds things, it saves its findings to a DB table. 

### Things I've run into which are slightly puzzling to me

The one step in this process that has always vexed me is the part where the
video gets broken apart into individual images for each frame. If you look back
in the git history for this repo, you'll notice that I was using OpenCV to
extract the frames at first. The problem I had with that was that, for my
videos in particular, OpenCV would end up not extracting all of the frames in
an individual file. This seems to have something to do with the video codec
that is used to encode the files on my little security camera system doesn't
have very good metadata so any open source tool just gets garbage in. 

I did find that `ffmpeg` could extract all the frames if you gave it the
correct incantations but it was always limited since you can only run it in one
process. Even the hardware acceleration that you can get with the Nvidia
Toolkit doesn't really help much (it seems to be more for encoding videos). I
tried breaking the work of extracting frames up into smaller chunks and then
just telling a bunch of workers to have `ffmpeg` only extract a particular
chunk but the problem then became the fact that, as the processing got farther
and farther into the video, `ffmpeg` would have to scan the video farther and
farther which was just a non-starter for very large videos. 

I then found `decord` which definitely does things a whole lot faster but has
the same limitations as OpenCV (since under the hood it seems to be using quite
a lot of the same primitives). Even though it's faster, it still seems like it
cani't quite keep up with the rate with which the process that is trying to see
if there are animals in the frames is working (especially if you're running
this on more than one machine). I'm guessing this is probably partly
because of the same thing I ran into with `ffmpeg`: scanning large videos
takes time and partly because I don't have a machine with 20 GPUs dedicated
to extracting images from videos so, even if I parallelize that process,
it's still fighting for resources with the detection process. Especially
with large videos, `decord` does seem to fail in weird ways when I
parallelize the process of extracting frames (which seem related to running out
of memory on the GPU?) So, that's where this project currently sits: a faster
but imperfect solution. Which doesn't quite feel right to me. Hopefully I'll
get some more time to work on this before I run out of room to store all my
security camera videos. All I want is to stare at cute little animals in my
backyard!

### A coyote walking through my backyard in the middle of the night

![](coyote.gif)
