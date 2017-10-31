# Album Art

Determining an album's genre based on its cover art. I downloaded 10,000 album covers across 10 genres, and split them 70/30 train/test.

I've built three models:

* a simple model which computes the colour histogram for each image, and uses Annoy to get a given test image's nearest neighbours in 512-d space.

* a random forest trained on the colour histograms.

* a CNN, using the VGG16 net's weights with the top 4 dense layers re-trained.

Interestingly the CNN doesn't outperform the colour models by very much, it seems most of the genre information is contained in the colour distributions. This sort of makes sense I think, intra-class variability in the kind of objects and strucures in the images probably out weighs inter-class differences.

![confusion](confusion.png) 
 
