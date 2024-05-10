/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/


#include "precomp.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <opencv/highgui.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vector>
#include "..\\include\\opencv2\\para_config.h"
#include "..\\include\\opencv2\\grey_assited.h"
using namespace std;



 /**
   * \brief check whether machine is little endian
   */
int littleendian()
{
    int intval = 1;
    uchar* uval = (uchar*)&intval;
    return uval[0] == 1;
}

/**
  * \brief save disparity to pfm format
  * 1-band PFM image, see http://netpbm.sourceforge.net/doc/pfm.html
  */
void writeFilePFM(float* data, int width, int height, char* filename, float scalefactor)
{
    // Open the file
    FILE* stream;
    errno_t error = fopen_s(&stream, filename, "wb");
    if (error != 0) {
        fprintf(stderr, "WriteFilePFM: could not open %s\n", filename);
        exit(1);
    }

    // sign of scalefact indicates endianness, see pfms specs
    if (littleendian())
        scalefactor = -scalefactor;

    // write the header: 3 lines: Pf, dimensions, scale factor (negative val == little endian)
    fprintf(stream, "Pf\n%d %d\n%f\n", width, height, scalefactor);

    int n = width;
    // write rows -- pfm stores rows in inverse order!
    for (int y = height - 1; y >= 0; y--) {
        float* ptr = data + y * width;
        // change invalid pixels (which seem to be represented as -10) to INF
        for (int x = 0; x < width; x++) {
            if (ptr[x] < 0)
                ptr[x] = INFINITY;
        }
        if ((int)fwrite(ptr, sizeof(float), n, stream) != n) {
            fprintf(stderr, "WriteFilePFM: problem writing data\n");
            exit(1);
        }
    }

    // close file
    fclose(stream);
}

namespace cv {
namespace phase_unwrapping {
class CV_EXPORTS_W HistogramPhaseUnwrapping_Impl : public HistogramPhaseUnwrapping
{
public:
    // Constructor
    explicit HistogramPhaseUnwrapping_Impl( const HistogramPhaseUnwrapping::Params &parameters =
                                            HistogramPhaseUnwrapping::Params() );
    // Destructor
    virtual ~HistogramPhaseUnwrapping_Impl() CV_OVERRIDE {};

    // Unwrap phase map
    void unwrapPhaseMap( InputArray wrappedPhaseMap, OutputArray unwrappedPhaseMap,
                         InputArray shadowMask = noArray() ) CV_OVERRIDE;
    // Get reliability map computed from the wrapped phase map
    void getInverseReliabilityMap( OutputArray reliabilityMap ) CV_OVERRIDE;

private:
    // Class describing a pixel
    class Pixel
    {
    private:
        // Value from the wrapped phase map
        float phaseValue;
        // Id of a pixel. Computed from its position in the Mat
        int idx;
        // Pixel is valid if it's not in a shadow region
        bool valid;
        // "Quality" parameter. See reference paper
        float inverseReliability;
        // Number of 2pi  that needs to be added to the pixel to unwrap the phase map
        int increment;
        // Number of pixels that are in the same group as the current pixel
        int nbrOfPixelsInGroup;
        // Group id. At first, group id is the same value as idx
        int groupId;
        // Pixel is alone in its group
        bool singlePixelGroup;
    public:
        Pixel();
        Pixel( float pV, int id, bool v, float iR, int inc );
        float getPhaseValue();
        int getIndex();
        bool getValidity();
        float getInverseReliability();
        int getIncrement();
        int getNbrOfPixelsInGroup();
        int getGroupId();
        bool getSinglePixelGroup();
        void setIncrement( int inc );
        // When a pixel which is not in a single group is added to a new group, we need to keep the previous increment and add "inc" to it.
        void changeIncrement( int inc );
        void setNbrOfPixelsInGroup( int nbr );
        void setGroupId( int gId );
        void setSinglePixelGroup( bool s );
    };
    // Class describing an Edge as presented in the reference paper
    class Edge
    {
    private:
        // Id of the first pixel that forms the edge
        int pixOneId;
        // Id of the second pixel that forms the edge
        int pixTwoId;
        // Number of 2pi that needs to be added to the second pixel to remove discontinuities
        int increment;
    public:
        Edge();
        Edge( int p1, int p2, int inc );
        int getPixOneId();
        int getPixTwoId();
        int getIncrement();
    };
    // Class describing a bin from the histogram
    class HistogramBin
    {
    private:
        float start;
        float end;
        std::vector<Edge> edges;
    public:
        HistogramBin();
        HistogramBin( float s, float e );
        void addEdge( Edge e );
        std::vector<Edge> getEdges();
    };
    // Class describing the histogram. Bins before "thresh" are smaller than the one after "thresh" value
    class Histogram
    {
    private:
        std::vector<HistogramBin> bins;
        float thresh;
        float smallWidth;
        float largeWidth;
        int nbrOfSmallBins;
        int nbrOfLargeBins;
        int nbrOfBins;
    public:
        Histogram();
        void createBins( float t, int nbrOfBinsBeforeThresh, int nbrOfBinsAfterThresh );
        void addBin( HistogramBin b );
        void addEdgeInBin( Edge e, int binIndex);
        float getThresh();
        float getSmallWidth();
        float getLargeWidth();
        int getNbrOfBins();
        std::vector<Edge> getEdgesFromBin( int binIndex );
    };
    // Params for phase unwrapping
    Params params;
    // Pixels from the wrapped phase map
    std::vector<Pixel> pixels;
    // Pixels from the unwrapped phase map
    std::vector<Pixel> unpixels;
    // Histogram used to unwrap
    Histogram histogram;
    // Compute pixel reliability.
    void computePixelsReliability( InputArray wrappedPhaseMap, InputArray shadowMask = noArray());
    // Compute edges reliability and sort them in the histogram
    void computeEdgesReliabilityAndCreateHistogram();
    // Methods that is used in the previous one
    void createAndSortEdge( int idx1, int idx2 );
    // Unwrap the phase map thanks to the histogram
    void unwrapHistogram();
    // add right number of 2*pi to the pixels
    void addIncrement(InputArray shadowMask, OutputArray unwrappedPhaseMap);
    // Gamma function from the paper
    float wrap( float a, float b );
    // Similar to the previous one but returns the number of 2pi that needs to be added
    int findInc( float a, float b );
};
// Default parameters
HistogramPhaseUnwrapping::Params::Params(){
    width = 800;
    height = 600;
    histThresh = static_cast<float>(3 * CV_PI * CV_PI);
    nbrOfSmallBins = 10;
    nbrOfLargeBins = 5;
}
HistogramPhaseUnwrapping_Impl::HistogramPhaseUnwrapping_Impl(
                            const HistogramPhaseUnwrapping::Params &parameters ) : params(parameters)
{

}

HistogramPhaseUnwrapping_Impl::Pixel::Pixel()
{

}
// Constructor
HistogramPhaseUnwrapping_Impl::Pixel::Pixel( float pV, int id, bool v, float iR, int inc )
{
    phaseValue = pV;
    idx = id;
    valid = v;
    inverseReliability = iR;
    increment = inc;
    nbrOfPixelsInGroup = 1;
    groupId = id;
    singlePixelGroup = true;
}

float HistogramPhaseUnwrapping_Impl::Pixel::getPhaseValue()
{
    return phaseValue;
}

int HistogramPhaseUnwrapping_Impl::Pixel::getIndex()
{
    return idx;
}

bool HistogramPhaseUnwrapping_Impl::Pixel::getValidity()
{
    return valid;
}

float HistogramPhaseUnwrapping_Impl::Pixel::getInverseReliability()
{
    return inverseReliability;
}

int HistogramPhaseUnwrapping_Impl::Pixel::getIncrement()
{
    return increment;
}

int HistogramPhaseUnwrapping_Impl::Pixel::getNbrOfPixelsInGroup()
{
    return nbrOfPixelsInGroup;
}

int HistogramPhaseUnwrapping_Impl::Pixel::getGroupId()
{
    return groupId;
}

bool HistogramPhaseUnwrapping_Impl::Pixel::getSinglePixelGroup()
{
    return singlePixelGroup;
}

void HistogramPhaseUnwrapping_Impl::Pixel::setIncrement( int inc )
{
    increment = inc;
}
/* When a pixel of a non-single group is added to an other non-single group, we need to add a new
increment to the one that was there previously and that was already removing some wraps.
*/
void HistogramPhaseUnwrapping_Impl::Pixel::changeIncrement( int inc )
{
    increment += inc;
}

void HistogramPhaseUnwrapping_Impl::Pixel::setNbrOfPixelsInGroup( int nbr )
{
    nbrOfPixelsInGroup = nbr;
}
void HistogramPhaseUnwrapping_Impl::Pixel::setGroupId( int gId )
{
    groupId = gId;
}

void HistogramPhaseUnwrapping_Impl::Pixel::setSinglePixelGroup( bool s )
{
    singlePixelGroup = s;
}

HistogramPhaseUnwrapping_Impl::Edge::Edge()
{

}
// Constructor
HistogramPhaseUnwrapping_Impl::Edge::Edge( int p1, int p2, int inc )
{
    pixOneId = p1;
    pixTwoId = p2;
    increment = inc;
}

int HistogramPhaseUnwrapping_Impl::Edge::getPixOneId()
{
    return pixOneId;
}

int HistogramPhaseUnwrapping_Impl::Edge::getPixTwoId()
{
    return pixTwoId;
}

int HistogramPhaseUnwrapping_Impl::Edge::getIncrement()
{
    return increment;
}

HistogramPhaseUnwrapping_Impl::HistogramBin::HistogramBin()
{

}

HistogramPhaseUnwrapping_Impl::HistogramBin::HistogramBin( float s, float e )
{
    start = s;
    end = e;
}

void HistogramPhaseUnwrapping_Impl::HistogramBin::addEdge( Edge e )
{
    edges.push_back(e);
}
std::vector<HistogramPhaseUnwrapping_Impl::Edge> HistogramPhaseUnwrapping_Impl::HistogramBin::getEdges()
{
    return edges;
}
HistogramPhaseUnwrapping_Impl::Histogram::Histogram()
{

}
/*
 * create histogram bins. Bins size is not uniform, as in the reference paper
 *
 */
void HistogramPhaseUnwrapping_Impl::Histogram::createBins( float t, int nbrOfBinsBeforeThresh,
                                                           int nbrOfBinsAfterThresh )
{
    thresh = t;

    nbrOfSmallBins = nbrOfBinsBeforeThresh;
    nbrOfLargeBins = nbrOfBinsAfterThresh;
    nbrOfBins = nbrOfBinsBeforeThresh + nbrOfBinsAfterThresh;

    smallWidth = thresh / nbrOfSmallBins;
    largeWidth = static_cast<float>(32 * CV_PI * CV_PI - thresh) / static_cast<float>(nbrOfLargeBins);

    for( int i = 0; i < nbrOfSmallBins; ++i )
    {
        addBin(HistogramBin(i * smallWidth, ( i + 1 ) * smallWidth));
    }
    for( int i = 0; i < nbrOfLargeBins; ++i )
    {
        addBin(HistogramBin(thresh + i * largeWidth, thresh + ( i + 1 ) * largeWidth));
    }
}
// Add a bin b to the histogram
void HistogramPhaseUnwrapping_Impl::Histogram::addBin( HistogramBin b )
{
    bins.push_back(b);
}
// Add edge E in bin binIndex
void HistogramPhaseUnwrapping_Impl::Histogram::addEdgeInBin( Edge e, int binIndex )
{
    bins[binIndex].addEdge(e);
}
float HistogramPhaseUnwrapping_Impl::Histogram::getThresh()
{
    return thresh;
}

float HistogramPhaseUnwrapping_Impl::Histogram::getSmallWidth()
{
    return smallWidth;
}

float HistogramPhaseUnwrapping_Impl::Histogram::getLargeWidth()
{
    return largeWidth;
}

int HistogramPhaseUnwrapping_Impl::Histogram::getNbrOfBins()
{
    return nbrOfBins;
}

std::vector<HistogramPhaseUnwrapping_Impl::Edge> HistogramPhaseUnwrapping_Impl::
                                                 Histogram::getEdgesFromBin( int binIndex )
{
    std::vector<HistogramPhaseUnwrapping_Impl::Edge> temp;
    temp = bins[binIndex].getEdges();
    return temp;
}
/* Method in which reliabilities are computed and edges are sorted in the histogram.
Increments are computed for each pixels.
 */
void HistogramPhaseUnwrapping_Impl::unwrapPhaseMap( InputArray wrappedPhaseMap,
                                                    OutputArray unwrappedPhaseMap,
                                                    InputArray shadowMask )
{
    Mat &wPhaseMap = *(Mat*) wrappedPhaseMap.getObj();
    Mat mask;
    int rows = params.height;
    int cols = params.width;
    if( shadowMask.empty() )
    {
        mask.create(rows, cols, CV_8UC1);
        mask = Scalar::all(255);
    }
    else
    {
        Mat &temp = *(Mat*) shadowMask.getObj();
        temp.copyTo(mask);
    }

    computePixelsReliability(wPhaseMap, mask);
    computeEdgesReliabilityAndCreateHistogram();

    unwrapHistogram();
    addIncrement(mask,unwrappedPhaseMap);
}

//compute pixels reliabilities according to "A novel algorithm based on histogram processing of reliability for two-dimensional phase unwrapping"

void HistogramPhaseUnwrapping_Impl::computePixelsReliability(InputArray wrappedPhaseMap,
    InputArray shadowMask)
{
    int rows = params.height;
    int cols = params.width;

    Mat& wPhaseMap = *(Mat*)wrappedPhaseMap.getObj();
    Mat& mask = *(Mat*)shadowMask.getObj();

    int idx; //idx is used to store pixel position (idx = i*cols + j)
    bool valid;//tells if a pixel is in the valid mask region

    // H, V, D1, D2 are from the paper
    float H, V, D1, D2, D;
    /* used to store neighbours coordinates
     * ul = upper left, um = upper middle, ur = upper right
     * ml = middle left, mr = middle right
     * ll = lower left, lm = lower middle, lr = lower right
     */
    Point ul, um, ur, ml, mr, ll, lm, lr;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (mask.at<uchar>(i, j) != 0) //if pixel is in a valid region
            {
                if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1)
                {
                    idx = i * cols + j;
                    valid = true;
                    Pixel p(wPhaseMap.at<float>(i, j), idx, valid,
                        static_cast<float>(16 * CV_PI * CV_PI), 0);
                    pixels.push_back(p);
                }
                else
                {
                    ul = Point(j-1, i-1);
                    um = Point(j, i-1);
                    ur = Point(j+1, i-1);
                    ml = Point(j-1, i);
                    mr = Point(j+1, i);
                    ll = Point(j-1, i+1);
                    lm = Point(j, i+1);
                    lr = Point(j+1, i+1);

                    Mat neighbourhood = mask( Rect( j-1, i-1, 3, 3 ) );
                    Scalar meanValue = mean(neighbourhood);

                    /* if mean value is different from 255, it means that one of the neighbouring
                     * pixel is not valid -> pixel (i,j) is considered as being on the border.
                     */
                    if( meanValue[0] != 255 )
                    {
                        idx = i * cols + j;
                        valid = true;
                        Pixel p(wPhaseMap.at<float>(i, j), idx, valid,
                                static_cast<float>(16 * CV_PI * CV_PI), 0);
                        pixels.push_back(p);
                    }
                    else
                    {
                        H = wrap(wPhaseMap.at<float>(ml.y, ml.x), wPhaseMap.at<float>(i, j))
                            - wrap(wPhaseMap.at<float>(i, j), wPhaseMap.at<float>(mr.y, mr.x));
                        V = wrap(wPhaseMap.at<float>(um.y, um.x), wPhaseMap.at<float>(i, j))
                            - wrap(wPhaseMap.at<float>(i, j), wPhaseMap.at<float>(lm.y, lm.x));
                        D1 = wrap(wPhaseMap.at<float>(ul.y, ul.x), wPhaseMap.at<float>(i, j))
                            - wrap(wPhaseMap.at<float>(i, j), wPhaseMap.at<float>(lr.y, lr.x));
                        D2 = wrap(wPhaseMap.at<float>(ur.y, ur.x), wPhaseMap.at<float>(i, j))
                            - wrap(wPhaseMap.at<float>(i, j), wPhaseMap.at<float>(ll.y, ll.x));
                        D = H * H + V * V + D1 * D1 + D2 * D2;

                        idx = i * cols + j;
                        valid = true;
                        Pixel p(wPhaseMap.at<float>(i, j), idx, valid, D, 0);
                        pixels.push_back(p);
                    }
                }
            }
            else // pixel is not in a valid region. It's inverse reliability is set to the maximum
            {
                idx = i * cols + j;
                valid = false;
                Pixel p(wPhaseMap.at<float>(i, j), idx, valid,
                        static_cast<float>(16 * CV_PI * CV_PI), 0);
                pixels.push_back(p);
            }
        }
    }
}

/* Edges are created from the vector of pixels. We loop on the vector and create the edges
 * that link the current pixel to his right neighbour (first edge) and the one that is under it (second edge)
 */
void HistogramPhaseUnwrapping_Impl::computeEdgesReliabilityAndCreateHistogram()
{
    int row;
    int col;
    histogram.createBins(params.histThresh, params.nbrOfSmallBins, params.nbrOfLargeBins);
    int nbrOfPixels = static_cast<int>(pixels.size());
    /* Edges are built by considering a pixel and it's right-neighbour and lower-neighbour.
     We discard non-valid pixels here.
     */
    for( int i = 0; i < nbrOfPixels; ++i )
    {
        if( pixels[i].getValidity() )
        {
            row = pixels[i].getIndex() / params.width;
            col = pixels[i].getIndex() % params.width;

            if( row != params.height - 1 && col != params.width -1 )
            {
                int idxRight, idxDown;
                idxRight = row * params.width + col + 1; // Pixel to the right
                idxDown = ( row + 1 ) * params.width + col; // Pixel under pixel i.
                createAndSortEdge(i, idxRight);
                createAndSortEdge(i, idxDown);
            }
            else if( row != params.height - 1 && col == params.width - 1 )
            {
                int idxDown = ( row + 1 ) * params.width + col;
                createAndSortEdge(i, idxDown);
            }
            else if( row == params.height - 1 && col != params.width - 1 )
            {
                int idxRight = row * params.width + col + 1;
                createAndSortEdge(i, idxRight);
            }
        }
    }
}
/*used along the previous method to sort edges in the histogram*/
void HistogramPhaseUnwrapping_Impl::createAndSortEdge( int idx1, int idx2 )
{
    if( pixels[idx2].getValidity() )
    {
        float edgeReliability = pixels[idx1].getInverseReliability() +
                                pixels[idx2].getInverseReliability();
        int inc = findInc(pixels[idx2].getPhaseValue(), pixels[idx1].getPhaseValue());
        Edge e(idx1, idx2, inc);

        if( edgeReliability < histogram.getThresh() )
        {
            int binIndex = static_cast<int> (ceil(edgeReliability / histogram.getSmallWidth()) - 1);
            if( binIndex == -1 )
            {
                binIndex = 0;
            }
            histogram.addEdgeInBin(e, binIndex);
        }
        else
        {
            int binIndex = params.nbrOfSmallBins +
                           static_cast<int> (ceil((edgeReliability - histogram.getThresh()) /
                                 histogram.getLargeWidth()) - 1);
            histogram.addEdgeInBin(e, binIndex);
        }
    }
}

void HistogramPhaseUnwrapping_Impl::unwrapHistogram()
{
    int nbrOfPixels = static_cast<int>(pixels.size());
    int nbrOfBins = histogram.getNbrOfBins();
    /* This vector is used to keep track of the number of pixels in each group and avoid useless group.
       For example, if lastPixelAddedToGroup[10] is equal to 5, it means that pixel "5" was the last one
       to be added to group 10. So, pixel "5" is the only one that has the correct value for parameter
       "numberOfPixelsInGroup" in order to avoid a loop on all the pixels to update this number*/
    std::vector<int> lastPixelAddedToGroup(nbrOfPixels, 0);
    for( int i = 0; i < nbrOfBins; ++i )
    {
        std::vector<Edge> currentEdges = histogram.getEdgesFromBin(i);
        int nbrOfEdgesInBin = static_cast<int>(currentEdges.size());

        for( int j = 0; j < nbrOfEdgesInBin; ++j )
        {

            int pOneId = currentEdges[j].getPixOneId();
            int pTwoId = currentEdges[j].getPixTwoId();
            // Both pixels are in a single group.
            if( pixels[pOneId].getSinglePixelGroup() && pixels[pTwoId].getSinglePixelGroup() )
            {
                float invRel1 = pixels[pOneId].getInverseReliability();
                float invRel2 = pixels[pTwoId].getInverseReliability();
                // Quality of pixel 2 is better than that of pixel 1 -> pixel 1 is added to group 2
                if( invRel1 > invRel2 )
                {
                    int newGroupId = pixels[pTwoId].getGroupId();
                    int newInc = pixels[pTwoId].getIncrement() + currentEdges[j].getIncrement();
                    pixels[pOneId].setGroupId(newGroupId);
                    pixels[pOneId].setIncrement(newInc);
                    lastPixelAddedToGroup[newGroupId] = pOneId; // Pixel 1 is the last one to be added to group 2
                }
                else
                {
                    int newGroupId = pixels[pOneId].getGroupId();
                    int newInc = pixels[pOneId].getIncrement() - currentEdges[j].getIncrement();
                    pixels[pTwoId].setGroupId(newGroupId);
                    pixels[pTwoId].setIncrement(newInc);
                    lastPixelAddedToGroup[newGroupId] = pTwoId;
                }
                pixels[pOneId].setNbrOfPixelsInGroup(2);
                pixels[pTwoId].setNbrOfPixelsInGroup(2);
                pixels[pOneId].setSinglePixelGroup(false);
                pixels[pTwoId].setSinglePixelGroup(false);
            }
            //p1 is in a single group, p2 is not -> p1 added to p2
            else if( pixels[pOneId].getSinglePixelGroup() && !pixels[pTwoId].getSinglePixelGroup() )
            {
                int newGroupId = pixels[pTwoId].getGroupId();
                int lastPix = lastPixelAddedToGroup[newGroupId];
                int newNbrOfPixelsInGroup = pixels[lastPix].getNbrOfPixelsInGroup() + 1;
                int newInc = pixels[pTwoId].getIncrement() + currentEdges[j].getIncrement();

                pixels[pOneId].setGroupId(newGroupId);
                pixels[pOneId].setNbrOfPixelsInGroup(newNbrOfPixelsInGroup);
                pixels[pTwoId].setNbrOfPixelsInGroup(newNbrOfPixelsInGroup);
                pixels[pOneId].setIncrement(newInc);
                pixels[pOneId].setSinglePixelGroup(false);

                lastPixelAddedToGroup[newGroupId] = pOneId;
            }
            //p2 is in a single group, p1 is not -> p2 added to p1
            else if( !pixels[pOneId].getSinglePixelGroup() && pixels[pTwoId].getSinglePixelGroup() )
            {
                int newGroupId = pixels[pOneId].getGroupId();
                int lastPix = lastPixelAddedToGroup[newGroupId];
                int newNbrOfPixelsInGroup = pixels[lastPix].getNbrOfPixelsInGroup() + 1;
                int newInc = pixels[pOneId].getIncrement() - currentEdges[j].getIncrement();

                pixels[pTwoId].setGroupId(newGroupId);
                pixels[pTwoId].setNbrOfPixelsInGroup(newNbrOfPixelsInGroup);
                pixels[pOneId].setNbrOfPixelsInGroup(newNbrOfPixelsInGroup);
                pixels[pTwoId].setIncrement(newInc);
                pixels[pTwoId].setSinglePixelGroup(false);

                lastPixelAddedToGroup[newGroupId] = pTwoId;
            }
            //p1 and p2 are in two different groups
            else if( pixels[pOneId].getGroupId() != pixels[pTwoId].getGroupId() )
            {
                int pOneGroupId = pixels[pOneId].getGroupId();
                int pTwoGroupId = pixels[pTwoId].getGroupId();

                float invRel1 = pixels[pOneId].getInverseReliability();
                float invRel2 = pixels[pTwoId].getInverseReliability();

                int lastAddedToGroupOne = lastPixelAddedToGroup[pOneGroupId];
                int lastAddedToGroupTwo = lastPixelAddedToGroup[pTwoGroupId];

                int nbrOfPixelsInGroupOne = pixels[lastAddedToGroupOne].getNbrOfPixelsInGroup();
                int nbrOfPixelsInGroupTwo = pixels[lastAddedToGroupTwo].getNbrOfPixelsInGroup();

                int totalNbrOfPixels = nbrOfPixelsInGroupOne + nbrOfPixelsInGroupTwo;

                if( nbrOfPixelsInGroupOne < nbrOfPixelsInGroupTwo ||
                   (nbrOfPixelsInGroupOne == nbrOfPixelsInGroupTwo && invRel1 >= invRel2) ) //group p1 added to group p2
                {
                    pixels[pTwoId].setNbrOfPixelsInGroup(totalNbrOfPixels);
                    pixels[pOneId].setNbrOfPixelsInGroup(totalNbrOfPixels);
                    int inc = pixels[pTwoId].getIncrement() + currentEdges[j].getIncrement() -
                                 pixels[pOneId].getIncrement();
                    lastPixelAddedToGroup[pTwoGroupId] = pOneId;

                    for( int k = 0; k < nbrOfPixels; ++k )
                    {
                        if( pixels[k].getGroupId() == pOneGroupId )
                        {
                            pixels[k].setGroupId(pTwoGroupId);
                            pixels[k].changeIncrement(inc);
                        }
                    }
                }
                else if( nbrOfPixelsInGroupOne > nbrOfPixelsInGroupTwo ||
                        (nbrOfPixelsInGroupOne == nbrOfPixelsInGroupTwo && invRel2 > invRel1) ) //group p2 added to group p1
                {
                    int oldGroupId = pTwoGroupId;
                    pixels[pOneId].setNbrOfPixelsInGroup(totalNbrOfPixels);
                    pixels[pTwoId].setNbrOfPixelsInGroup(totalNbrOfPixels);
                    int inc = pixels[pOneId].getIncrement() - currentEdges[j].getIncrement() -
                              pixels[pTwoId].getIncrement();
                    lastPixelAddedToGroup[pOneGroupId] = pTwoId;

                    for( int k = 0; k < nbrOfPixels; ++k )
                    {
                        if( pixels[k].getGroupId() == oldGroupId )
                        {
                            pixels[k].setGroupId(pOneGroupId);
                            pixels[k].changeIncrement(inc);
                        }
                    }
                }
            }
        }
    }
}

float getA(InputArray arcs, int n)
{
    Mat& arcs1 = *(Mat*)arcs.getObj();
    float ans = 0;
    if (n == 1)
    {
        /*return arcs[0][0];*/
    }
    
    Mat temp = Mat::zeros(3, 3, CV_32FC1);
    int i3, j3, k3;/*
    float ans;*/
    for (i3 = 0; i3 < n; i3++)
    {
        for (j3 = 0; j3 < n - 1; j3++)
        {
            for (k3 = 0; k3 < n - 1; k3++)
            {
                temp.at<float>(j3,k3) = arcs1.at<float>(j3 + 1,(k3 >= i3) ? k3 + 1 : k3);

            }
        }
        float t = getA(temp, n - 1);
        if (i3 % 2 == 0)
        {
            ans += arcs1.at<float>(0,i3) * t;
        }
        else
        {
            ans -= arcs1.at<float>(0,i3) * t;
        }
    }
    return ans;
}

void  getAStart(InputArray arcs, int n, float ans[3][3])
{
    Mat& arcs1 = *(Mat*)arcs.getObj();
    if (n == 1)
    {
        ans[0][0] = 1;
        return;
    }
    int i4, j4, k4, t;
    Mat temp = Mat::zeros(3,3,CV_32FC1);
    for (i4 = 0; i4 < n; i4++)
    {
        for (j4 = 0; j4 < n; j4++)
        {
            for (k4 = 0; k4 < n - 1; k4++)
            {
                for (t = 0; t < n - 1; t++)
                {
                    temp.at<float>(k4,t) = arcs1.at<float>(k4 >= i4 ? k4 + 1 : k4,t >= j4 ? t + 1 : t);
                }
            }


            ans[j4][i4] = getA(temp, n - 1);  //此处顺便进行了转置
            if ((i4 + j4) % 2 == 1)
            {
                ans[j4][i4] = -ans[j4][i4];
            }
        }
    }
}

bool GetMatrixInverse(InputArray src, OutputArray des)
{
    Mat& src1 = *(Mat*)src.getObj();
    Mat& des1 = *(Mat*)des.getObj();
    int n = 3;
    float flag = getA(src1, n);
    float t[3][3];
    if (flag == 0)
    {
        return false;
    }
    else
    {
        getAStart(src1, n, t);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                des1.at<float>(i,j) = t[i][j] / flag;
            }

        }
    }
    return true;
}

/* use the grey img  unwrapped the phase */



/* use the grey img  unwrapped the phase */






//void HistogramPhaseUnwrapping_Impl::addIncrement(InputArray shadowMask, OutputArray unwrappedPhaseMap)
//{
//    Mat& uPhaseMap = *(Mat*)unwrappedPhaseMap.getObj();
//    Mat uHeightMap;
//    Mat uHeightMap8;
//    int samples_size[3];
//    int rows = params.height;
//    int cols = params.width;
//    int height = params.height;
//    int width = params.width;
//    Mat mask;
//    if (shadowMask.empty())
//    {
//        mask.create(rows, cols, CV_8UC1);
//        mask = Scalar::all(255);
//    }
//    else
//    {
//        Mat& temp = *(Mat*)shadowMask.getObj();
//        temp.copyTo(mask);
//    }
//    pcl::PointXYZ pointxyz;
//    //
//    samples_size[1] = rows;
//    samples_size[2] = cols;
//    samples_size[0] = 10000;/*
//    Mat m_3 = cv::Mat::zeros(3, samples_size, CV_8UC1,Scalar::all(0));*/
//    //
//
//    if (uPhaseMap.empty())
//    {
//        uPhaseMap.create(rows, cols, CV_32FC1);
//        uPhaseMap = Scalar::all(0);
//    }
//    if (uHeightMap.empty())
//    {/*
//        uHeightMap = cv::Mat(3, samples_size, CV_8UC1, Scalar::all(0));*/
//        uHeightMap.create(height,width, CV_32FC1);
//        uHeightMap = Scalar::all(0);
//    }
//    int nbrOfPixels = static_cast<int>(pixels.size());
//
//    ///// 显示点云
//    //pcl::visualization::CloudViewer viewer("Cloud Viewer");
//    pcl::PointCloud<pcl::PointXYZ> cloud;
//    //将上面的数据填入点云
//    cloud.width = width;//设置点云宽度
//    cloud.height = height; //设置点云高度
//    cloud.is_dense = false; //非密集型
//    cloud.points.resize(cloud.width * cloud.height); //变形，无序
//        //设置这些点的坐标
//    //for (size_t i = 0; i < cloud.points.size(); ++i)
//    //{
//    //    cloud.points[i].x =
//    //        cloud.points[i].y =
//    //        cloud.points[i].z = uPhaseMap.at<float>(row, col) * 6.5 /*mm*/ / static_cast<float>(2 * CV_PI);
//    //}
//    //保存到PCD文件
//    
//
//    for (int i = 0; i < nbrOfPixels; ++i)
//    {
//        int row = pixels[i].getIndex() / params.width;
//        int col = pixels[i].getIndex() % params.width;
//        /*int z = 10000;*/
//        pcl::PointXYZ point;
//        if (pixels[i].getValidity())
//        {
//            uPhaseMap.at<float>(row, col) = pixels[i].getPhaseValue() +
//                static_cast<float>(2 * CV_PI * pixels[i].getIncrement());
//            //uHeightMap.at<float>(row, col) = uPhaseMap.at<float>(row, col) * 6.5/*mm*//
//            //    static_cast<float>(2 * CV_PI);
//            uHeightMap.at<float>(row, col) = uPhaseMap.at<float>(row, col) * 6.5/*mm*/ /
//                static_cast<float>(2 * CV_PI);// compute the z coordinate
//            //point.x = row;
//            //point.y = col;
//            //point.z = uPhaseMap.at<float>(row, col) * 6.5/*mm*/ / static_cast<float>(2 * CV_PI);// compute the z coordinate;;
//            cloud.points[i].x = row;
//            cloud.points[i].y = col;
//            cloud.points[i].z = uPhaseMap.at<float>(row, col) * 65/*mm*//static_cast<float>(2 * CV_PI);// compute the z coordinate;
//            //cloud.points.push_back(point);
//
//                
//        }
//
//    }
//    uHeightMap.convertTo(uHeightMap8, CV_8U, 1, 128);
//    if (!uHeightMap8.empty()) {
//        std::string height_file = "E:\\vs_engineering\\structured_light\\structured_light\\13.pfm";
//        cv::imwrite("E:\\vs_engineering\\structured_light\\structured_light\\13_uHeightMap8.png", uHeightMap8);/*
//        writeFilePFM((float*)(uHeightMap8.data), width, height, (char*)(height_file.c_str()), 1.0);*/
//    }
//        
//
//    /*use another method unnwrap phase */
//    int m, n;
//    int count = 0;
//    int k1, k2;
//    float med_x;
//    Mat win_x = Mat::zeros(Size(25, 1), CV_32FC1);
//    Mat win_xx = Mat::zeros(Size(25, 1), CV_32FC1);
//    float cam_intrinsic[3][3] = { {1630.67666736282, 0, 647.0104444320734},{0, 1627.92270280131, 500.2543765201615},{0, 0, 1} };
//    float proj_intrinsic[3][3] = { {1308.937860572184, 0, 408.4427608534915},{0, 2512.223625783619, 899.3388049088476},{0, 0, 1} };
//    float kc_cam[5] = { -0.1044199952532041, 0.174840207031923, 0.0008839620840498505, 0.0005872317437199291, 0 };
//    float kc_pro[5] = { 0.08946727771800242, -0.08985477805218943, 0.01660318277654648, -0.03213903144260157, 0 };
//    int cam_rotation[3][3] = { {1,0,0,},{0,1,0},{0,0,1} };
//    int cam_translation[3] = { 0,0,1 };
//    float proj_rotation[3][3] = { {0.9875662380306558, -0.02347651004271293, 0.1554405963002994}, {0.04431923199472351, 0.9902565232716074, -0.1320144832714439} ,{-0.1508268251252944, 0.1372620544587718, 0.9789843702676673} };
//    float proj_translation[3] = { -151.2254362243205, 10.70163440085295, 101.534677062764 };
//    int k, l;
//    Mat /*R[3][3], T[3] = { 0, 0, 0 }*/ T1;
//
//     /*distort the undistort_cam*/
//     Mat undistort_cam;
//     Mat cameraMatrix1 = Mat(3, 3, CV_64FC1, Scalar::all(0)); /* 摄像机内参数矩阵 */
//     Mat distCoeffs1 = Mat(1, 5, CV_64FC1, Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2 */
//     Mat cameraMatrix2 = Mat(3, 3, CV_64FC1, Scalar::all(0)); /* 投影仪内参数矩阵 */
//     Mat distCoeffs2 = Mat(1, 5, CV_64FC1, Scalar::all(0)); /* 投影仪的5个畸变系数：k1,k2,p1,p2 */
//     FileStorage fs("C:/Users/DELL/OneDrive/桌面/zby_code/output/outputs/parameter.xml", FileStorage::READ);
//     FileStorage fs_p("C:/Users/DELL/OneDrive/桌面/zby_code/output/outputs/undistort_point.xml", FileStorage::READ);
//     Mat R, T;
//
//     if (fs.isOpened())
//     {
//         fs["cameraMatrix1"] >> cameraMatrix1;
//         distCoeffs1 = fs["distCoeffs1"].mat();
//         cameraMatrix2 = fs["cameraMatrix2"].mat();
//         distCoeffs2 = fs["distCoeffs2"].mat();
//         R = fs["R"].mat();
//         T = fs["T"].mat();
//     }
//     else
//     {
//         cout << "Can't open parameter.xml" << endl;
//     }
//
//     if (fs_p.isOpened())
//     {
//         fs_p["undistort_cam"] >> undistort_cam;
//     }
//     else
//     {
//         cout << "Can't open parameter.xml" << endl;
//     }
//     Mat RT1/*[3][4]*/;
//     Mat RT2/*[3][4]*/;
//     fs.release(); fs_p.release();
//     Mat temp4 = Mat::zeros(3, 1, CV_64FC1);
//     hconcat(cameraMatrix1, temp4, RT1);
//     Mat temp5;
//     vconcat(R.t(), T.t(), temp5);
//     RT2 = cameraMatrix2 * temp5.t();
//     /*distort the undistort_cam*/
//    //for (k = 0; k < 3; k++)
//    //{
//    //    for (l = 0; l < 3; l++)
//    //    {
//
//    //        //R[k][l] = (cam_rotation[k][l] == 0 ? 0:( proj_rotation[k][l] / cam_rotation[k][l])) ;
//    //        R.at<float>(k,l) = proj_rotation[k][l] ;
//    //        T1.at<float>(k) =T1.at<float>(k)+ R.at<float>(k,l) * cam_translation[l];
//    //    }
//    //    T.at<float>(k) = proj_translation[k] - T1.at<float>(k);
//    //}
//
//    /*
//
//    R = proj_rotation / cam_rotation;
//    T = proj_translation - R * cam_translation;*/
//    float cc_pro[2] = { proj_intrinsic[0][2],proj_intrinsic[1][2] };
//    float cc_cam[2] = { cam_intrinsic[0][2],cam_intrinsic[1][2] };
//    float f_p[2] = { proj_intrinsic[0][0],proj_intrinsic[1][1] };
//    float f_c[2] = { cam_intrinsic[0][0], cam_intrinsic[1][1] };
//
//    /*矩阵的乘法RT1*/
//    //Mat RT1(3,4,CV_32FC1);
//    //int a[12] = { 1,0,0,0,0,1,0,0,0,0,1,0 };
//    //    /*
//    //double C1[3][4] = { {1,0,0,0},{0,1,0,0},{0,0,1,0} };*/
//    //Mat C1(3, 4, CV_32FC1,a);
//
//    //for (int i1 = 0; i1 < 3; i1++)
//    //{
//    //    for (int j1 = 0; j1 < 4; j1++)
//    //    {
//    //        RT1[i1][j1] = C1.at<float>(0,j1) * cam_intrinsic[i1][0]+ C1.at<float>(1,j1) * cam_intrinsic[i1][1] + C1.at<float>(2,j1) * cam_intrinsic[i1][2];
//    //        // 3*4
//    //    }
//    //}
//    /*矩阵的乘法RT1*/
//    /***********/
//    /*矩阵的乘法RT2*/
//    //Mat R_1(3, 3, CV_32FC1);
//    //Mat T_1(1, 3, CV_32FC1);
//    //Mat C2(3, 4, CV_32FC1);
//    //float R_1[3][3];
//    //float T_1[3];
//    //float C2[3][4];
//
//    //T_1[0] = T.at<float>(0); T_1[1] = T.at<float>(1); T_1[2] = T.at<float>(2);
//    //R_1[0][0] = R.at<float>(0,0); R_1[1][0] = R.at<float>(0,1); R_1[2][0] = R.at<float>(0,2);
//    //R_1[0][1] = R.at<float>(1,0); R_1[1][1] = R.at<float>(1,1); R_1[2][1] = R.at<float>(1,2);
//    //R_1[0][2] = R.at<float>(2,0); R_1[1][2] = R.at<float>(2,1); R_1[2][2] = R.at<float>(2,2);
//
//    //C2[0][0] = R_1[0][0]; C2[1][1] = R_1[1][1]; C2[2][2] = R_1[2][2];
//    //C2[0][1] = R_1[1][0]; C2[0][2] = R_1[2][0];
//    //C2[1][0] = R_1[0][1]; C2[1][2] = R_1[2][1];
//    //C2[2][1] = R_1[1][2]; C2[2][0] = R_1[0][2];
//    //C2[1][3] = T_1[1]; C2[0][3] = T_1[0]; C2[2][3] = T_1[2];
//
//    Mat mask8;
//
//    ////Mat RT2(3,4,CV_32FC1);
//    //float RT2[3][4];
//    //for (int i2 = 0; i2 < 3; i2++)
//    //{
//    //    for (int j2 = 0; j2 < 4; j2++)
//    //    {
//    //        RT2[i2][j2] = C2[0][j2] * proj_intrinsic[i2][0] + C2[1][j2] * proj_intrinsic[i2][1] + C2[2][j2] * proj_intrinsic[i2][2];
//    //        // 3*4
//    //    }
//    //}
//    /*Mat Mc(3,4,CV_32FC1);
//    Mat Mp(3,4,CV_32FC1);
//    Mc.at<float>(0,0) = RT1[0][0]; Mc.at<float>(0,1) = RT1[0][1]; Mc.at<float>(0,2) = RT1[0][2]; Mc.at<float>(0,3) = RT1[0][3];
//    Mc.at<float>(1,0) = RT1[1][0]; Mc.at<float>(1,1) = RT1[1][1]; Mc.at<float>(1,2) = RT1[1][2]; Mc.at<float>(1,3) = RT1[1][3];
//    Mc.at<float>(2,0) = RT1[2][0]; Mc.at<float>(2,1) = RT1[2][1]; Mc.at<float>(2,2) = RT1[2][2]; Mc.at<float>(2,3) = RT1[2][3];
//
//    Mp.at<float>(0,0) = RT2[0][0]; Mp.at<float>(0,1) = RT2[0][1]; Mp.at<float>(0,2) = RT2[0][2]; Mp.at<float>(0,3) = RT2[0][3];
//    Mp.at<float>(1,0) = RT2[1][0]; Mp.at<float>(1,1) = RT2[1][1]; Mp.at<float>(1,2) = RT2[1][2]; Mp.at<float>(1,3) = RT2[1][3];
//    Mp.at<float>(2,0) = RT2[2][0]; Mp.at<float>(2,1) = RT2[2][1]; Mp.at<float>(2,2) = RT2[2][2]; Mp.at<float>(2,3) = RT2[2][3];*/
//    /*矩阵乘法*/
//
//    /*矩阵的乘法RT2*/
//    /*ending***************************/
//
//    /*restruct the 3D poinnt*/
//    bool** D = new bool* [height];
//    double** Pro_point = new double* [height];
//    for (int i = 0; i < height; i++)
//    {
//        D[i] = new bool[width];
//        Pro_point[i] = new double[width];//height个指针，指向weidth个指针。内存着投影仪的点坐标
//    }
//    int image_count = 1;
//
//    char path[100] = "C:\\Users\\DELL\\OneDrive\\桌面\\zby_code\\decodeimage\\";
//    char path1[100];
//    char sub_name[20];
//    strcpy(path1, path);
//    sprintf(sub_name, "%d_%d.bmp", 1, single_image_serial);
//    strcat(path1, sub_name);
//    Mat image = imread(path, IMREAD_GRAYSCALE);
//    image.convertTo(image, CV_64FC1);
//
//
//    //decode_gp_single(D, Pro_point, 1, path/*,phase1*/);
//    for (int m = 2; m < height-2; m++)
//    {
//        for (int n = 2; n < width-2; n++)
//        {
//            if (D[m][n] == 1)
//            {
//                //cout << mask.at<unsigned char>(m, n);
//
//                count = 0;
//                for (k1 = -2; k1 < 3; k1++)
//                {
//                    for (k2 = -2; k2 < 3; k2++)
//                    {
//                        //if (mask.at<unsigned char>(m + k1, n + k2) != 0)
//                        if (D[m + k1][n + k2] != 0)
//                        {
//                            win_x.at<float>(count) = uPhaseMap.at<float>(m + k1, n + k2);
//                            count = count + 1;
//                        }
//                    }
//                }
//                cv::sort(win_x, win_xx, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
//                med_x = win_xx.at<float>(floor(count / 2 + 1));
//                if ((uPhaseMap.at<float>(m, n) - med_x) > CV_PI)
//                {
//                    uPhaseMap.at<float>(m, n) = uPhaseMap.at<float>(m, n) - 2 * CV_PI;
//                }
//                else if ((uPhaseMap.at<float>(m, n) - med_x) < -CV_PI)
//                {
//                    uPhaseMap.at<float>(m, n) = uPhaseMap.at<float>(m, n) + 2 * CV_PI;
//                }
//
//            }  
//            else {
//                uPhaseMap.at<float>(m, n) = 0.;
//            }
//        }
//    }
//
//    int i3, j3;
//    float proj_j;
//    Mat A(3, 3, CV_32FC1);
//    Mat A_1(3, 3, CV_32FC1);
//    Mat B(3, 1, CV_32FC1);
//    int pointsize = 0; pcl::PointCloud<pcl::PointXYZ> cloud2;
//    //将上面的数据填入点云
//    cloud2.width = width;//设置点云宽度
//    cloud2.height = height; //设置点云高度
//    cloud2.is_dense = false; //非密集型
//    cloud2.points.resize(long(cloud2.width* cloud2.height)); //变形，无序
//    
//    String maskpath;
//    mask.convertTo(mask8, CV_8U, 1, 128);
//    maskpath = "E://vs_engineering//structured_light//structured_light//";
//    imwrite(maskpath + "mask.png", mask8);
//    //for (j3 = 3; j3 <= height - 3; j3++)
//    //{
//    //    for (i3 = 3; i3 <= width - 3; i3++)
//    //    {
//    //        if (mask.at<unsigned char>(j3, i3) == 255)
//    //        {
//    //            proj_j = uPhaseMap.at<float>(j3, i3);
//    //            D[0][0] = Mc[0][0] - Mc[2][0] * i3;
//    //            D[0][1] = Mc[0][1] - Mc[2][1] * i3;
//    //            D[0][2] = Mc[0][2] - Mc[2][2] * i3;
//
//    //            D[1][0] = Mc[1][0] - Mc[2][0] * j3;
//    //            D[1][1] = Mc[1][1] - Mc[2][1] * j3;
//    //            D[1][2] = Mc[1][2] - Mc[2][2] * j3;
//
//    //            D[2][0] = Mp[0][0] - Mp[2][0] * proj_j;
//    //            D[2][1] = Mp[0][1] - Mp[2][1] * proj_j;
//    //            D[2][2] = Mp[0][2] - Mp[2][2] * proj_j;
//
//    //            E[0] = Mc[2][3] * i3 - Mc[0][3];
//    //            E[1] = Mc[2][3] * j3 - Mc[1][3];
//    //            E[2] = Mp[2][3] * proj_j - Mp[0][3];
//
//    //            if (E[0] == 0)
//    //            {
//    //                if (E[1] == 0)
//    //                {
//    //                    cloud2.points[pointsize].x = D[0][2] * E[2];
//    //                    cloud2.points[pointsize].y = D[1][2] * E[2];
//    //                    cloud2.points[pointsize].z = D[2][2] * E[2];
//    //                    pointsize = pointsize + 1;
//    //                }
//    //                else if (E[2] == 0)
//    //                {
//    //                    cloud2.points[pointsize].x = D[0][1] * E[1];
//    //                    cloud2.points[pointsize].y = D[1][1] * E[1];
//    //                    cloud2.points[pointsize].z = D[2][1] * E[1];
//    //                    pointsize = pointsize + 1;
//    //                }
//    //            }
//    //            else
//    //            {
//    //                cloud2.points[pointsize].x = D[0][0] * E[0];
//    //                cloud2.points[pointsize].y = D[1][0] * E[0];
//    //                cloud2.points[pointsize].z = D[2][0] * E[0];
//    //                pointsize = pointsize + 1;
//    //            }
//
//
//
//    //        }
//    //    }
//    //}
//    
//
//    
//
//    
//    // D is the mask
//    // Pro_point is the uwappedphase
//    //k_image is the decode image count numbers
//    // path1 is the decode imgae path 
//
//
//
// 
//
//
//
//
//    for (i3 = 50; i3 <= height - 50; i3++)
//    {
//        for (j3 = 50; j3 <= width - 50; j3++)
//        {
//            //if (mask.at<unsigned char>(j3, i3) == 255)
//            if (D[i3][j3] == 1)
//            {
//
//                Point2d point1 = undistort_cam.at<Vec2d>(i3, j3); //distort the point
//
//                double cam_i = point1.y;
//                double cam_j = point1.x;
//                //double proj_j = Pro_point[i3][j3];
//                double proj_j = uPhaseMap.at<float>(i3, j3);
//
//                A.at<float>(0, 0)  = RT1.at<double>(0, 0) - RT1.at<double>(2, 0) * cam_j;
//                A.at<float>(0, 1) = RT1.at<double>(0, 1) - RT1.at<double>(2, 1) * cam_j;
//                A.at<float>(0, 2) = RT1.at<double>(0, 2) - RT1.at<double>(2, 2) * cam_j;                                                         
//                A.at<float>(1, 0) = RT1.at<double>(1, 0) - RT1.at<double>(2, 0) * cam_i;
//                A.at<float>(1, 1) = RT1.at<double>(1, 1) - RT1.at<double>(2, 1) * cam_i;
//                A.at<float>(1, 2) = RT1.at<double>(1, 2) - RT1.at<double>(2, 2) * cam_i;
//                                                         
//                A.at<float>(2, 0) = RT2.at<double>(0, 0) - RT2.at<double>(2, 0) * proj_j;
//                A.at<float>(2, 1) = RT2.at<double>(0, 1) - RT2.at<double>(2, 1) * proj_j;
//                A.at<float>(2, 2) = RT2.at<double>(0, 2) - RT2.at<double>(2, 2) * proj_j;
//
//                B.at<float>(0, 0) = RT1.at<double>(2, 3) * cam_j  - RT1.at<double>(0, 3);
//                B.at<float>(1, 0) = RT1.at<double>(2, 3) * cam_i - RT1.at<double>(1, 3);
//                B.at<float>(2, 0) = RT2.at<double>(2, 3) * proj_j - RT2.at<double>(0, 3);
//
//                //得到给定矩阵src的逆矩阵保存到des中。//按第一行展开计算|A|
//
//                Mat A_t = A.t();
//                Mat d_point = ((A_t * A).inv()) * A_t * B;
//                //GetMatrixInverse(A, A_1);
//                //Mat d_point = (A_1)*B;
//
//                
//
//                /*cloud2.points[pointsize].x = D_1[0][2] * E[2] + D_1[0][1] * E[1] + D_1[0][0] * E[0];
//                cloud2.points[pointsize].y = D_1[1][2] * E[2] + D_1[1][1] * E[1] + D_1[1][0] * E[0];
//                cloud2.points[pointsize].z = D_1[2][2] * E[2] + D_1[2][1] * E[1] + D_1[2][0] * E[0];*/
//                cloud2.points[pointsize].x = d_point.at<float>(0, 0);
//                cloud2.points[pointsize].y = d_point.at<float>(1, 0);
//                cloud2.points[pointsize].z = d_point.at<float>(2, 0);
//                pointsize = pointsize + 1;
//
//               /*else if (E[2] == 0)
//                {
//                    cloud2.points[pointsize].x = D_1[0][1] * E[1];
//                    cloud2.points[pointsize].y = D_1[1][1] * E[1];
//                    cloud2.points[pointsize].z = D_1[2][1] * E[1];
//                    pointsize = pointsize + 1;
//                }*/
//            }
//        }
//    }
//
//    /*restruct the 3D poinnt*/
//
//    pcl::io::savePCDFileASCII("E:\\vs_engineering\\structured_light\\structured_light\\test_pcd_1.pcd", cloud);
//    pcl::io::savePCDFileASCII("E:\\vs_engineering\\structured_light\\structured_light\\test_pcd_2.pcd", cloud2); //将点云保存到PCD文件中
//    std::cout << "pcd has finished and saved!!!! "<< endl;
//    std::cerr << "Saved " << cloud.points.size() << " data points to test_pcd.pcd." << std::endl;
//    ////显示点云数据
//    //for (size_t i = 0; i < cloud.points.size(); ++i)
//    //    std::cerr << "    " << cloud.points[i].x << " " << cloud.points[i].y << " " << cloud.points[i].z << std::endl;
//    
//    
//}
void HistogramPhaseUnwrapping_Impl::addIncrement(InputArray shadowMask, OutputArray unwrappedPhaseMap)
{
    Mat& uPhaseMap = *(Mat*)unwrappedPhaseMap.getObj();
    int rows = params.height;
    int cols = params.width;
    if (uPhaseMap.empty())
    {
        uPhaseMap.create(rows, cols, CV_32FC1);
        uPhaseMap = Scalar::all(0);
    }
    int nbrOfPixels = static_cast<int>(pixels.size());
    for (int i = 0; i < nbrOfPixels; ++i)
    {
        int row = pixels[i].getIndex() / params.width;
        int col = pixels[i].getIndex() % params.width;

        if (pixels[i].getValidity())
        {
            uPhaseMap.at<float>(row, col) = pixels[i].getPhaseValue() +
                static_cast<float>(2 * CV_PI * pixels[i].getIncrement());
        }
    }
}
float HistogramPhaseUnwrapping_Impl::wrap( float a, float b )
{
    float result;
    float difference = a - b;
    float pi = static_cast<float>(CV_PI);
    if( difference > pi )
        result = ( difference - 2 * pi );
    else if( difference < -pi )
        result = ( difference + 2 * pi );
    else
        result = difference;
    return result;
}

int HistogramPhaseUnwrapping_Impl::findInc( float a, float b )
{
    float difference;
    int wrapValue;
    difference = b - a;
    float pi = static_cast<float>(CV_PI);
    if( difference > pi )
        wrapValue = -1;
    else if( difference < -pi )
        wrapValue = 1;
    else
        wrapValue = 0;
    return wrapValue;
}

//create a Mat that shows pixel inverse reliabilities
void HistogramPhaseUnwrapping_Impl::getInverseReliabilityMap( OutputArray inverseReliabilityMap )
{
    int rows = params.height;
    int cols = params.width;
    Mat &reliabilityMap_ = *(Mat*) inverseReliabilityMap.getObj();
    if( reliabilityMap_.empty() )
        reliabilityMap_.create(rows, cols, CV_32FC1);
    for( int i = 0; i < rows; ++i )
    {
        for( int j = 0; j < cols; ++j )
        {
            int idx = i * cols + j;
            reliabilityMap_.at<float>(i, j) = pixels[idx].getInverseReliability();
        }
    }
}

Ptr<HistogramPhaseUnwrapping> HistogramPhaseUnwrapping::create( const HistogramPhaseUnwrapping::Params
                                                                &params )
{
    return makePtr<HistogramPhaseUnwrapping_Impl>(params);
}

}
}
