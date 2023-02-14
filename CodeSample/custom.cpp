#include "custom.hpp"
#include <vector>

#ifdef DEBUG_CHESSBOARD
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#define DPRINTF(...)  CV_LOG_INFO(NULL, cv::format("calib3d: " __VA_ARGS__))
#else
#define DPRINTF(...)
#endif

using namespace cv;
using namespace std;

/***************************************************************************************************/
//COMPUTE INTENSITY HISTOGRAM OF INPUT IMAGE
template<typename ArrayContainer>
static void icvGetIntensityHistogram256(const Mat& img, ArrayContainer& piHist)
{
    for (int i = 0; i < 256; i++)
        piHist[i] = 0;
    // sum up all pixel in row direction and divide by number of columns
    for (int j = 0; j < img.rows; ++j)
    {
        const uchar* row = img.ptr<uchar>(j);
        for (int i = 0; i < img.cols; i++)
        {
            piHist[row[i]]++;
        }
    }
}
/***************************************************************************************************/
//SMOOTH HISTOGRAM USING WINDOW OF SIZE 2*iWidth+1
template<int iWidth_, typename ArrayContainer>
static void icvSmoothHistogram256(const ArrayContainer& piHist, ArrayContainer& piHistSmooth, int iWidth = 0)
{
    CV_DbgAssert(iWidth_ == 0 || (iWidth == iWidth_ || iWidth == 0));
    iWidth = (iWidth_ != 0) ? iWidth_ : iWidth;
    CV_Assert(iWidth > 0);
    CV_DbgAssert(piHist.size() == 256);
    CV_DbgAssert(piHistSmooth.size() == 256);
    for (int i = 0; i < 256; ++i)
    {
        int iIdx_min = std::max(0, i - iWidth);
        int iIdx_max = std::min(255, i + iWidth);
        int iSmooth = 0;
        for (int iIdx = iIdx_min; iIdx <= iIdx_max; ++iIdx)
        {
            CV_DbgAssert(iIdx >= 0 && iIdx < 256);
            iSmooth += piHist[iIdx];
        }
        piHistSmooth[i] = iSmooth/(2*iWidth+1);
    }
}

/***************************************************************************************************/
//COMPUTE FAST HISTOGRAM GRADIENT
template<typename ArrayContainer>
static void icvGradientOfHistogram256(const ArrayContainer& piHist, ArrayContainer& piHistGrad)
{
    CV_DbgAssert(piHist.size() == 256);
    CV_DbgAssert(piHistGrad.size() == 256);
    piHistGrad[0] = 0;
    int prev_grad = 0;
    for (int i = 1; i < 255; ++i)
    {
        int grad = piHist[i-1] - piHist[i+1];
        if (std::abs(grad) < 100)
        {
            if (prev_grad == 0)
                grad = -100;
            else
                grad = prev_grad;
        }
        piHistGrad[i] = grad;
        prev_grad = grad;
    }
    piHistGrad[255] = 0;
}

/***************************************************************************************************/
//PERFORM SMART IMAGE THRESHOLDING BASED ON ANALYSIS OF INTENSTY HISTOGRAM
static void icvBinarizationHistogramBased(Mat & img)
{
    CV_Assert(img.channels() == 1 && img.depth() == CV_8U);
    int iCols = img.cols;
    int iRows = img.rows;
    int iMaxPix = iCols*iRows;
    int iMaxPix1 = iMaxPix/100;
    const int iNumBins = 256;
    const int iMaxPos = 20;
    cv::AutoBuffer<int, 256> piHistIntensity(iNumBins);
    cv::AutoBuffer<int, 256> piHistSmooth(iNumBins);
    cv::AutoBuffer<int, 256> piHistGrad(iNumBins);
    cv::AutoBuffer<int> piMaxPos(iMaxPos);

    icvGetIntensityHistogram256(img, piHistIntensity);

#if 0
    // get accumulated sum starting from bright
    cv::AutoBuffer<int, 256> piAccumSum(iNumBins);
    piAccumSum[iNumBins-1] = piHistIntensity[iNumBins-1];
    for (int i = iNumBins - 2; i >= 0; --i)
    {
        piAccumSum[i] = piHistIntensity[i] + piAccumSum[i+1];
    }
#endif

    // first smooth the distribution
    //const int iWidth = 1;
    icvSmoothHistogram256<1>(piHistIntensity, piHistSmooth);

    // compute gradient
    icvGradientOfHistogram256(piHistSmooth, piHistGrad);

    // check for zeros
    unsigned iCntMaxima = 0;
    for (int i = iNumBins-2; (i > 2) && (iCntMaxima < iMaxPos); --i)
    {
        if ((piHistGrad[i-1] < 0) && (piHistGrad[i] > 0))
        {
            int iSumAroundMax = piHistSmooth[i-1] + piHistSmooth[i] + piHistSmooth[i+1];
            if (!(iSumAroundMax < iMaxPix1 && i < 64))
            {
                piMaxPos[iCntMaxima++] = i;
            }
        }
    }

    DPRINTF("HIST: MAXIMA COUNT: %d (%d, %d, %d, ...)", iCntMaxima,
            iCntMaxima > 0 ? piMaxPos[0] : -1,
            iCntMaxima > 1 ? piMaxPos[1] : -1,
            iCntMaxima > 2 ? piMaxPos[2] : -1);

    int iThresh = 0;

    CV_Assert((size_t)iCntMaxima <= piMaxPos.size());

    DPRINTF("HIST: MAXIMA COUNT: %d (%d, %d, %d, ...)", iCntMaxima,
                iCntMaxima > 0 ? piMaxPos[0] : -1,
                iCntMaxima > 1 ? piMaxPos[1] : -1,
                iCntMaxima > 2 ? piMaxPos[2] : -1);

    if (iCntMaxima == 0)
    {
        // no any maxima inside (only 0 and 255 which are not counted above)
        // Does image black-write already?
        const int iMaxPix2 = iMaxPix / 2;
        for (int sum = 0, i = 0; i < 256; ++i) // select mean intensity
        {
            sum += piHistIntensity[i];
            if (sum > iMaxPix2)
            {
                iThresh = i;
                break;
            }
        }
    }
    else if (iCntMaxima == 1)
    {
        iThresh = piMaxPos[0]/2;
    }
    else if (iCntMaxima == 2)
    {
        iThresh = (piMaxPos[0] + piMaxPos[1])/2;
    }
    else // iCntMaxima >= 3
    {
        // CHECKING THRESHOLD FOR WHITE
        int iIdxAccSum = 0, iAccum = 0;
        for (int i = iNumBins - 1; i > 0; --i)
        {
            iAccum += piHistIntensity[i];
            // iMaxPix/18 is about 5,5%, minimum required number of pixels required for white part of chessboard
            if ( iAccum > (iMaxPix/18) )
            {
                iIdxAccSum = i;
                break;
            }
        }

        unsigned iIdxBGMax = 0;
        int iBrightMax = piMaxPos[0];
        // printf("iBrightMax = %d\n", iBrightMax);
        for (unsigned n = 0; n < iCntMaxima - 1; ++n)
        {
            iIdxBGMax = n + 1;
            if ( piMaxPos[n] < iIdxAccSum )
            {
                break;
            }
            iBrightMax = piMaxPos[n];
        }

        // CHECKING THRESHOLD FOR BLACK
        int iMaxVal = piHistIntensity[piMaxPos[iIdxBGMax]];

        //IF TOO CLOSE TO 255, jump to next maximum
        if (piMaxPos[iIdxBGMax] >= 250 && iIdxBGMax + 1 < iCntMaxima)
        {
            iIdxBGMax++;
            iMaxVal = piHistIntensity[piMaxPos[iIdxBGMax]];
        }

        for (unsigned n = iIdxBGMax + 1; n < iCntMaxima; n++)
        {
            if (piHistIntensity[piMaxPos[n]] >= iMaxVal)
            {
                iMaxVal = piHistIntensity[piMaxPos[n]];
                iIdxBGMax = n;
            }
        }

        //SETTING THRESHOLD FOR BINARIZATION
        int iDist2 = (iBrightMax - piMaxPos[iIdxBGMax])/2;
        iThresh = iBrightMax - iDist2;
        DPRINTF("THRESHOLD SELECTED = %d, BRIGHTMAX = %d, DARKMAX = %d", iThresh, iBrightMax, piMaxPos[iIdxBGMax]);
    }

    if (iThresh > 0)
    {
        img = (img >= iThresh);
    }
}

#ifdef DEBUG_CHESSBOARD
static void SHOW(const std::string & name, Mat & img)
{
    imshow(name, img);
#if DEBUG_CHESSBOARD_TIMEOUT
    waitKey(DEBUG_CHESSBOARD_TIMEOUT);
#else
    while ((uchar)waitKey(0) != 'q') {}
#endif
}
static void SHOW_QUADS(const std::string & name, const Mat & img_, ChessBoardQuad * quads, int quads_count)
{
    Mat img = img_.clone();
    if (img.channels() == 1)
        cvtColor(img, img, COLOR_GRAY2BGR);
    for (int i = 0; i < quads_count; ++i)
    {
        ChessBoardQuad & quad = quads[i];
        for (int j = 0; j < 4; ++j)
        {
            line(img, quad.corners[j]->pt, quad.corners[(j + 1) & 3]->pt, Scalar(0, 240, 0), 1, LINE_AA);
        }
    }
    imshow(name, img);
#if DEBUG_CHESSBOARD_TIMEOUT
    waitKey(DEBUG_CHESSBOARD_TIMEOUT);
#else
    while ((uchar)waitKey(0) != 'q') {}
#endif
}
#else
#define SHOW(...)
#define SHOW_QUADS(...)
#endif

/** This structure stores information about the chessboard corner.*/
struct ChessBoardCorner
{
    cv::Point2f pt;  // Coordinates of the corner
    int row;         // Board row index
    int count;       // Number of neighbor corners
    struct ChessBoardCorner* neighbors[4]; // Neighbor corners

    ChessBoardCorner(const cv::Point2f& pt_ = cv::Point2f()) :
        pt(pt_), row(0), count(0)
    {
        neighbors[0] = neighbors[1] = neighbors[2] = neighbors[3] = NULL;
    }

    float sumDist(int& n_) const
    {
        float sum = 0;
        int n = 0;
        for (int i = 0; i < 4; ++i)
        {
            if (neighbors[i])
            {
                sum += sqrt(normL2Sqr<float>(neighbors[i]->pt - pt));
                n++;
            }
        }
        n_ = n;
        return sum;
    }
};

/** This structure stores information about the chessboard quadrangle.*/
struct ChessBoardQuad
{
    int count;      // Number of quad neighbors
    int group_idx;  // quad group ID
    int row, col;   // row and column of this quad
    bool ordered;   // true if corners/neighbors are ordered counter-clockwise
    float edge_len; // quad edge len, in pix^2
    // neighbors and corners are synced, i.e., neighbor 0 shares corner 0
    ChessBoardCorner *corners[4]; // Coordinates of quad corners
    struct ChessBoardQuad *neighbors[4]; // Pointers of quad neighbors

    ChessBoardQuad(int group_idx_ = -1) :
        count(0),
        group_idx(group_idx_),
        row(0), col(0),
        ordered(0),
        edge_len(0)
    {
        corners[0] = corners[1] = corners[2] = corners[3] = NULL;
        neighbors[0] = neighbors[1] = neighbors[2] = neighbors[3] = NULL;
    }
};

class ChessBoardDetector
{
public:
    cv::Mat binarized_image;
    Size pattern_size;

    cv::AutoBuffer<ChessBoardQuad> all_quads;
    cv::AutoBuffer<ChessBoardCorner> all_corners;

    int all_quads_count;

    ChessBoardDetector(const Size& pattern_size_) :
        pattern_size(pattern_size_),
        all_quads_count(0)
    {
    }

    void reset()
    {
        all_quads.deallocate();
        all_corners.deallocate();
        all_quads_count = 0;
    }

    void generateQuads(const cv::Mat& image_, int flags);

    bool processQuads(std::vector<cv::Point2f>& out_corners, int &prev_sqr_size);

    void findQuadNeighbors();

    void findConnectedQuads(std::vector<ChessBoardQuad*>& out_group, int group_idx);

    int checkQuadGroup(std::vector<ChessBoardQuad*>& quad_group, std::vector<ChessBoardCorner*>& out_corners);

    int cleanFoundConnectedQuads(std::vector<ChessBoardQuad*>& quad_group);

    int orderFoundConnectedQuads(std::vector<ChessBoardQuad*>& quads);

    void orderQuad(ChessBoardQuad& quad, ChessBoardCorner& corner, int common);

#ifdef ENABLE_TRIM_COL_ROW
    void trimCol(std::vector<ChessBoardQuad*>& quads, int col, int dir);
    void trimRow(std::vector<ChessBoardQuad*>& quads, int row, int dir);
#endif

    int addOuterQuad(ChessBoardQuad& quad, std::vector<ChessBoardQuad*>& quads);

    void removeQuadFromGroup(std::vector<ChessBoardQuad*>& quads, ChessBoardQuad& q0);

    bool checkBoardMonotony(const std::vector<cv::Point2f>& corners);
};


// does a fast check if a chessboard is in the input image. This is a workaround to
// a problem of cvFindChessboardCorners being slow on images with no chessboard
// - src: input binary image
// - size: chessboard size
// Returns 1 if a chessboard can be in this image and findChessboardCorners should be called,
// 0 if there is no chessboard, -1 in case of error
int checkChessboardBinary(const cv::Mat & img, const cv::Size & size)
{
    CV_Assert(img.channels() == 1 && img.depth() == CV_8U);

    Mat white = img.clone();
    Mat black = img.clone();

    int result = 0;
    for ( int erosion_count = 0; erosion_count <= 3; erosion_count++ )
    {
        if ( 1 == result )
            break;

        if ( 0 != erosion_count ) // first iteration keeps original images
        {
            erode(white, white, Mat(), Point(-1, -1), 1);
            dilate(black, black, Mat(), Point(-1, -1), 1);
        }

        vector<pair<float, int> > quads;
        fillQuads(white, black, 128, 128, quads);
        if (checkQuads(quads, size))
            result = 1;
    }
    return result;
}

bool findChessboardCornersCustom(InputArray image_, Size pattern_size, OutputArray corners_, int flags)
{
    //CV_INSTRUMENT_REGION();  //Inutile dans notre cas, permet de faire des calculs de performances

    DPRINTF("==== findChessboardCorners(img=%dx%d, pattern=%dx%d, flags=%d)",
            image_.cols(), image_.rows(), pattern_size.width, pattern_size.height, flags);

    bool found = false;

    const int min_dilations = 0;
    const int max_dilations = 7;

    int type = image_.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    Mat img = image_.getMat();

    CV_CheckType(type, depth == CV_8U && (cn == 1 || cn == 3 || cn == 4),
            "Only 8-bit grayscale or color images are supported");

    if (pattern_size.width <= 2 || pattern_size.height <= 2)
        CV_Error(Error::StsOutOfRange, "Both width and height of the pattern should have bigger than 2");

    if (!corners_.needed())
        CV_Error(Error::StsNullPtr, "Null pointer to corners");

    std::vector<cv::Point2f> out_corners;

    if (img.channels() != 1)
    {
        cvtColor(img, img, COLOR_BGR2GRAY);
    }

    int prev_sqr_size = 0;

    Mat thresh_img_new = img.clone();
    icvBinarizationHistogramBased(thresh_img_new); // process image in-place
    SHOW("New binarization", thresh_img_new);

    if (flags & CALIB_CB_FAST_CHECK)
    {
        //perform new method for checking chessboard using a binary image.
        //image is binarised using a threshold dependent on the image histogram
        if (checkChessboardBinary(thresh_img_new, pattern_size) <= 0) //fall back to the old method
        {
            if (!checkChessboard(img, pattern_size))
            {
                corners_.release();
                return false;
            }
        }
    }

    ChessBoardDetector detector(pattern_size);

    // Try our standard "1" dilation, but if the pattern is not found, iterate the whole procedure with higher dilations.
    // This is necessary because some squares simply do not separate properly with a single dilation.  However,
    // we want to use the minimum number of dilations possible since dilations cause the squares to become smaller,
    // making it difficult to detect smaller squares.
    for (int dilations = min_dilations; dilations <= max_dilations; dilations++)
    {
        //USE BINARY IMAGE COMPUTED USING icvBinarizationHistogramBased METHOD
        dilate( thresh_img_new, thresh_img_new, Mat(), Point(-1, -1), 1 );

        // So we can find rectangles that go to the edge, we draw a white line around the image edge.
        // Otherwise FindContours will miss those clipped rectangle contours.
        // The border color will be the image mean, because otherwise we risk screwing up filters like cvSmooth()...
        rectangle( thresh_img_new, Point(0,0), Point(thresh_img_new.cols-1, thresh_img_new.rows-1), Scalar(255,255,255), 3, LINE_8);

        detector.reset();
        detector.generateQuads(thresh_img_new, flags);
        DPRINTF("Quad count: %d/%d", detector.all_quads_count, (pattern_size.width/2+1)*(pattern_size.height/2+1));
        SHOW_QUADS("New quads", thresh_img_new, &detector.all_quads[0], detector.all_quads_count);
        if (detector.processQuads(out_corners, prev_sqr_size))
        {
            found = true;
            break;
        }
    }

    DPRINTF("Chessboard detection result 0: %d", (int)found);

    // revert to old, slower, method if detection failed
    if (!found)
    {
        if (flags & CALIB_CB_NORMALIZE_IMAGE)
        {
            img = img.clone();
            equalizeHist(img, img);
        }

        Mat thresh_img;
        prev_sqr_size = 0;

        DPRINTF("Fallback to old algorithm");
        const bool useAdaptive = flags & CALIB_CB_ADAPTIVE_THRESH;
        if (!useAdaptive)
        {
            // empiric threshold level
            // thresholding performed here and not inside the cycle to save processing time
            double mean = cv::mean(img).val[0];
            int thresh_level = std::max(cvRound(mean - 10), 10);
            threshold(img, thresh_img, thresh_level, 255, THRESH_BINARY);
        }
        //if flag CALIB_CB_ADAPTIVE_THRESH is not set it doesn't make sense to iterate over k
        int max_k = useAdaptive ? 6 : 1;
        for (int k = 0; k < max_k && !found; k++)
        {
            for (int dilations = min_dilations; dilations <= max_dilations; dilations++)
            {
                // convert the input grayscale image to binary (black-n-white)
                if (useAdaptive)
                {
                    int block_size = cvRound(prev_sqr_size == 0
                                             ? std::min(img.cols, img.rows) * (k % 2 == 0 ? 0.2 : 0.1)
                                             : prev_sqr_size * 2);
                    block_size = block_size | 1;
                    // convert to binary
                    adaptiveThreshold( img, thresh_img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, block_size, (k/2)*5 );
                    if (dilations > 0)
                        dilate( thresh_img, thresh_img, Mat(), Point(-1, -1), dilations-1 );

                }
                else
                {
                    dilate( thresh_img, thresh_img, Mat(), Point(-1, -1), 1 );
                }
                SHOW("Old binarization", thresh_img);

                // So we can find rectangles that go to the edge, we draw a white line around the image edge.
                // Otherwise FindContours will miss those clipped rectangle contours.
                // The border color will be the image mean, because otherwise we risk screwing up filters like cvSmooth()...
                rectangle( thresh_img, Point(0,0), Point(thresh_img.cols-1, thresh_img.rows-1), Scalar(255,255,255), 3, LINE_8);

                detector.reset();
                detector.generateQuads(thresh_img, flags);
                DPRINTF("Quad count: %d/%d", detector.all_quads_count, (pattern_size.width/2+1)*(pattern_size.height/2+1));
                SHOW_QUADS("Old quads", thresh_img, &detector.all_quads[0], detector.all_quads_count);
                if (detector.processQuads(out_corners, prev_sqr_size))
                {
                    found = 1;
                    break;
                }
            }
        }
    }

    DPRINTF("Chessboard detection result 1: %d", (int)found);

    if (found)
        found = detector.checkBoardMonotony(out_corners);

    DPRINTF("Chessboard detection result 2: %d", (int)found);

    // check that none of the found corners is too close to the image boundary
    if (found)
    {
        const int BORDER = 8;
        for (int k = 0; k < pattern_size.width*pattern_size.height; ++k)
        {
            if( out_corners[k].x <= BORDER || out_corners[k].x > img.cols - BORDER ||
                out_corners[k].y <= BORDER || out_corners[k].y > img.rows - BORDER )
            {
                found = false;
                break;
            }
        }
    }

    DPRINTF("Chessboard detection result 3: %d", (int)found);

    if (found)
    {
        if ((pattern_size.height & 1) == 0 && (pattern_size.width & 1) == 0 )
        {
            int last_row = (pattern_size.height-1)*pattern_size.width;
            double dy0 = out_corners[last_row].y - out_corners[0].y;
            if (dy0 < 0)
            {
                int n = pattern_size.width*pattern_size.height;
                for(int i = 0; i < n/2; i++ )
                {
                    std::swap(out_corners[i], out_corners[n-i-1]);
                }
            }
        }
        cv::cornerSubPix(img, out_corners, Size(2, 2), Size(-1,-1),
                         cv::TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 15, 0.1));
    }

    Mat(out_corners).copyTo(corners_);
    return found;
}