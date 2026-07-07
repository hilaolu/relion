#include <catch2/catch.hpp>
#include "src/amyloid_finder.h"

#include <cmath>
#include <vector>

namespace
{
struct TestPsiScore
{
    MultidimArray<RFLOAT> signal;
    MultidimArray<RFLOAT> nonsignal;
};

void configureSyntheticFinder(AmyloidFinder &finder)
{
    finder.psi_step = 45.;
    finder.nr_psi = 4;
    finder.nr_threads = 1;
    finder.shift_step = 2;
    finder.ori_xsize = 24;
    finder.ori_ysize = 24;
    finder.down_xsize = 32;
    finder.down_ysize = 32;
    finder.large_box = 32;
    finder.crop_box = 32;
    finder.ilengthmax = 8;
    finder.iwidthmax = 4;
    finder.imin_signal = 1;
    finder.imax_signal = 2;
    finder.imin_nonsignal = 3;
    finder.imax_nonsignal = 3;
}

void buildPaddedMicrograph(const AmyloidFinder &finder, MultidimArray<RFLOAT> &image, MultidimArray<RFLOAT> &Mbig)
{
    Mbig.initZeros(finder.large_box, finder.large_box);
    Mbig.setXmippOrigin();
    for (long int i = STARTINGY(Mbig); i <= FINISHINGY(Mbig); i++)
    {
        long int ip = i;
        if (i < STARTINGY(image)) ip = 2 * STARTINGY(image) - i;
        else if (i > FINISHINGY(image)) ip = 2 * FINISHINGY(image) - i;

        for (long int j = STARTINGX(Mbig); j <= FINISHINGX(Mbig); j++)
        {
            long int jp = j;
            if (j < STARTINGX(image)) jp = 2 * STARTINGX(image) - j;
            else if (j > FINISHINGX(image)) jp = 2 * FINISHINGX(image) - j;

            A2D_ELEM(Mbig, i, j) = A2D_ELEM(image, ip, jp);
        }
    }
}

void calculateReferencePsiScore(
        AmyloidFinder &finder,
        const MultidimArray<RFLOAT> &rotated_img,
        int ipsi,
        std::vector<FourierTransformer> &transformers,
        TestPsiScore &out)
{
    MultidimArray<RFLOAT> scores_perline, nonscores_perline;
    scores_perline.initZeros(finder.crop_box, finder.crop_box);
    scores_perline.setXmippOrigin();
    nonscores_perline.initZeros(finder.crop_box, finder.crop_box);
    nonscores_perline.setXmippOrigin();
    out.signal.initZeros(finder.crop_box / finder.shift_step, finder.crop_box / finder.shift_step);
    out.signal.setXmippOrigin();
    out.nonsignal.initZeros(finder.crop_box / finder.shift_step, finder.crop_box / finder.shift_step);
    out.nonsignal.setXmippOrigin();

    MultidimArray<RFLOAT> oneline_tmp(finder.ilengthmax);
    for (int i = 0; i < finder.nr_threads; i++)
        transformers[i].setReal(oneline_tmp);

    int my_skip_side_length = finder.ilengthmax / 2;
    for (int ypos = my_skip_side_length; ypos < YSIZE(rotated_img) - my_skip_side_length; ypos += 1)
    {
        int cen_ypos = ypos - YSIZE(rotated_img) / 2;
        MultidimArray<RFLOAT> oneline(finder.ilengthmax);
        MultidimArray<Complex> FTline(finder.ilengthmax / 2 + 1);

        for (int xpos = my_skip_side_length; xpos < XSIZE(rotated_img) - my_skip_side_length; xpos += 1)
        {
            int cen_xpos = xpos - XSIZE(rotated_img) / 2;

            for (int iline = 0; iline < finder.ilengthmax; iline++)
                DIRECT_A1D_ELEM(oneline, iline) = A2D_ELEM(rotated_img, cen_ypos, cen_xpos + iline - finder.ilengthmax / 2);

            transformers[0].FourierTransform(oneline, FTline, false);

            for (int isig = finder.imin_signal; isig <= finder.imax_signal; isig++)
                A2D_ELEM(scores_perline, cen_ypos, cen_xpos) += norm(DIRECT_A1D_ELEM(FTline, isig));

            for (int isig = finder.imin_nonsignal; isig <= finder.imax_nonsignal; isig++)
                A2D_ELEM(nonscores_perline, cen_ypos, cen_xpos) += norm(DIRECT_A1D_ELEM(FTline, isig));
        }
    }

    int my_skip_side_width = finder.iwidthmax / 2;
    for (int ypos = 0; ypos < YSIZE(rotated_img) / 2 - my_skip_side_width; ypos += finder.shift_step)
    {
        for (int ipassy = 0; ipassy < 2; ipassy++)
        {
            int cen_ypos = (ipassy == 0) ? ypos : -ypos;
            if (ypos == 0 && ipassy == 1) continue;

            for (int xpos = 0; xpos < XSIZE(rotated_img) / 2 - my_skip_side_width; xpos += finder.shift_step)
            {
                for (int ipass = 0; ipass < 2; ipass++)
                {
                    int cen_xpos = (ipass == 0) ? xpos : -xpos;
                    if (xpos == 0 && ipass == 1) continue;

                    for (int iwidth = 0; iwidth < finder.iwidthmax; iwidth++)
                    {
                        A2D_ELEM(out.signal, cen_ypos / finder.shift_step, cen_xpos / finder.shift_step) +=
                                A2D_ELEM(scores_perline, cen_ypos + iwidth - finder.iwidthmax / 2, cen_xpos);
                        A2D_ELEM(out.nonsignal, cen_ypos / finder.shift_step, cen_xpos / finder.shift_step) +=
                                A2D_ELEM(nonscores_perline, cen_ypos + iwidth - finder.iwidthmax / 2, cen_xpos);
                    }
                }
            }
        }
    }

    RFLOAT psi = finder.getPsiAngle(ipsi);
    selfRotate(out.signal, -psi);
    selfRotate(out.nonsignal, -psi);
}

void referenceGetScore(AmyloidFinder &finder, MultidimArray<RFLOAT> &image, MultidimArray<RFLOAT> &Mscore,
                       MultidimArray<RFLOAT> &Mangle, RFLOAT &skew, RFLOAT &kurt)
{
    MultidimArray<RFLOAT> Mbig;
    buildPaddedMicrograph(finder, image, Mbig);

    std::vector<MultidimArray<RFLOAT> > rotated_imgs(finder.nr_psi);
    std::vector<FourierTransformer> transformers(finder.nr_threads);
    for (int ipsi = 0; ipsi < finder.nr_psi / 2; ipsi++)
    {
        RFLOAT psi = finder.getPsiAngle(ipsi);
        MultidimArray<RFLOAT> Mrot;
        Mrot.setXmippOrigin();
        Mrot.initZeros(finder.large_box, finder.large_box);
        rotate(Mbig, Mrot, psi, 'Z', true);

        MultidimArray<Complex > FT, FT2;
        transformers[0].FourierTransform(Mrot, FT, false);
        windowFourierTransform(FT, FT2, finder.crop_box);
        Mrot.resize(finder.crop_box, finder.crop_box);
        transformers[0].inverseFourierTransform(FT2, Mrot);
        Mrot.setXmippOrigin();
        rotated_imgs[ipsi] = Mrot;
    }

    MultidimArray<RFLOAT> Mzero(finder.crop_box, finder.crop_box);
    Mzero.setXmippOrigin();
    for (int ipsi = finder.nr_psi / 2; ipsi < finder.nr_psi; ipsi++)
    {
        rotated_imgs[ipsi] = Mzero;
        for (long int i = STARTINGY(Mzero) + 1; i <= FINISHINGY(Mzero) - 1; i++)
            for (long int j = STARTINGX(Mzero) + 1; j <= FINISHINGX(Mzero) - 1; j++)
                A2D_ELEM(rotated_imgs[ipsi], i, j) = A2D_ELEM(rotated_imgs[ipsi - finder.nr_psi / 2], -j, i);
    }

    std::vector<TestPsiScore> scores(finder.nr_psi);
    for (int ipsi = 0; ipsi < finder.nr_psi; ipsi++)
        calculateReferencePsiScore(finder, rotated_imgs[ipsi], ipsi, transformers, scores[ipsi]);

    Mangle.initZeros(finder.down_ysize / finder.shift_step, finder.down_xsize / finder.shift_step);
    Mangle.setXmippOrigin();
    Mscore.initZeros(Mangle);
    MultidimArray<RFLOAT> Msum, Mnonsum, Mnonscore, Mneighbour, Mneighbour2;
    Msum.initZeros(Mangle);
    Mnonsum.initZeros(Mangle);
    Mnonscore.initZeros(Mangle);
    Mneighbour.initZeros(Mangle);
    Mneighbour2.initZeros(Mangle);

    int xsize = XSIZE(Mscore);
    int ysize = YSIZE(Mscore);
    for (int ipsi = 0; ipsi < finder.nr_psi; ipsi++)
    {
        RFLOAT mypsi = finder.getPsiAngle(ipsi);
        for (int ypos = 0; ypos < ysize; ypos++)
        {
            int cen_ypos = ypos - ysize / 2;
            for (int xpos = 0; xpos < xsize; xpos++)
            {
                int cen_xpos = xpos - xsize / 2;
                RFLOAT myscore = A2D_ELEM(scores[ipsi].signal, cen_ypos, cen_xpos);
                RFLOAT mynonscore = A2D_ELEM(scores[ipsi].nonsignal, cen_ypos, cen_xpos);
                A2D_ELEM(Msum, cen_ypos, cen_xpos) += myscore;
                A2D_ELEM(Mnonsum, cen_ypos, cen_xpos) += mynonscore;

                if (myscore > A2D_ELEM(Mscore, cen_ypos, cen_xpos))
                {
                    A2D_ELEM(Mscore, cen_ypos, cen_xpos) = myscore;
                    A2D_ELEM(Mangle, cen_ypos, cen_xpos) = mypsi;
                    int ipsi_nb = (ipsi == 0) ? finder.nr_psi - 1 : ipsi - 1;
                    A2D_ELEM(Mneighbour, cen_ypos, cen_xpos) = A2D_ELEM(scores[ipsi_nb].signal, cen_ypos, cen_xpos);
                    ipsi_nb = (ipsi == finder.nr_psi - 1) ? 0 : ipsi + 1;
                    A2D_ELEM(Mneighbour, cen_ypos, cen_xpos) += A2D_ELEM(scores[ipsi_nb].signal, cen_ypos, cen_xpos);
                }

                if (mynonscore > A2D_ELEM(Mnonscore, cen_ypos, cen_xpos))
                {
                    A2D_ELEM(Mnonscore, cen_ypos, cen_xpos) = mynonscore;
                    int ipsi_nb = (ipsi == 0) ? finder.nr_psi - 1 : ipsi - 1;
                    A2D_ELEM(Mneighbour2, cen_ypos, cen_xpos) = A2D_ELEM(scores[ipsi_nb].nonsignal, cen_ypos, cen_xpos);
                    ipsi_nb = (ipsi == finder.nr_psi - 1) ? 0 : ipsi + 1;
                    A2D_ELEM(Mneighbour2, cen_ypos, cen_xpos) += A2D_ELEM(scores[ipsi_nb].nonsignal, cen_ypos, cen_xpos);
                }
            }
        }
    }

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Msum)
    {
        DIRECT_MULTIDIM_ELEM(Msum, n) -= DIRECT_MULTIDIM_ELEM(Mscore, n) + DIRECT_MULTIDIM_ELEM(Mneighbour, n);
        DIRECT_MULTIDIM_ELEM(Msum, n) /= (RFLOAT)(finder.nr_psi - 3);
        RFLOAT Zscore_signal = 0.;
        if (DIRECT_MULTIDIM_ELEM(Msum, n) > 0.)
            Zscore_signal = (DIRECT_MULTIDIM_ELEM(Mscore, n) - DIRECT_MULTIDIM_ELEM(Msum, n)) / DIRECT_MULTIDIM_ELEM(Msum, n);
        DIRECT_MULTIDIM_ELEM(Mscore, n) = Zscore_signal;

        DIRECT_MULTIDIM_ELEM(Mnonsum, n) -= DIRECT_MULTIDIM_ELEM(Mnonscore, n) + DIRECT_MULTIDIM_ELEM(Mneighbour2, n);
        DIRECT_MULTIDIM_ELEM(Mnonsum, n) /= (RFLOAT)(finder.nr_psi - 3);
        RFLOAT Zscore_nonsignal = 0.;
        if (DIRECT_MULTIDIM_ELEM(Mnonsum, n) > 0.)
        {
            Zscore_nonsignal = (DIRECT_MULTIDIM_ELEM(Mnonscore, n) - DIRECT_MULTIDIM_ELEM(Mnonsum, n)) / DIRECT_MULTIDIM_ELEM(Mnonsum, n);
            Zscore_nonsignal = (Zscore_nonsignal < 0.7) ? 0 : 1;
        }
        DIRECT_MULTIDIM_ELEM(Mnonscore, n) = Zscore_nonsignal;
    }

    Mnonscore = finder.growNonSignalMask(Mnonscore, finder.iwidthmax);

    RFLOAT sum = 0., sum2 = 0.;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Mscore)
    {
        DIRECT_MULTIDIM_ELEM(Mscore, n) *= (1. - DIRECT_MULTIDIM_ELEM(Mnonscore, n));
        sum += DIRECT_MULTIDIM_ELEM(Mscore, n);
        sum2 += DIRECT_MULTIDIM_ELEM(Mscore, n) * DIRECT_MULTIDIM_ELEM(Mscore, n);
    }

    RFLOAT n = NZYXSIZE(Msum);
    sum /= n;
    sum2 /= n;
    sum2 = sqrt(sum2 - sum * sum);
    skew = 0.;
    kurt = 0.;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Msum)
    {
        RFLOAT aux = (DIRECT_MULTIDIM_ELEM(Mscore, n) - sum) / sum2;
        skew += aux * aux * aux;
        kurt += aux * aux * aux * aux;
    }
    kurt *= n * (n + 1) / ((n - 1) * (n - 2) * (n - 3));
    skew *= n / ((n - 1) * (n - 2));
}
}

TEST_CASE("Amyloid FOM streaming path preserves legacy all-psi numerics", "[amyloid]")
{
    AmyloidFinder finder;
    configureSyntheticFinder(finder);

    MultidimArray<RFLOAT> image(finder.ori_ysize, finder.ori_xsize);
    image.setXmippOrigin();
    for (long int i = STARTINGY(image); i <= FINISHINGY(image); i++)
    {
        for (long int j = STARTINGX(image); j <= FINISHINGX(image); j++)
        {
            A2D_ELEM(image, i, j) =
                    0.7 * std::sin(0.31 * i + 0.17 * j) +
                    0.4 * std::cos(0.11 * i - 0.23 * j) +
                    ((i + 2 * j) % 5) * 0.03;
        }
    }

    MultidimArray<RFLOAT> reference_score, reference_angle, streaming_score, streaming_angle;
    RFLOAT reference_skew, reference_kurt, streaming_skew, streaming_kurt;

    referenceGetScore(finder, image, reference_score, reference_angle, reference_skew, reference_kurt);
    finder.getScoreForOneMicrograph(image, streaming_score, streaming_angle, streaming_skew, streaming_kurt, false);

    REQUIRE(SAME_SHAPE2D(reference_score, streaming_score));
    REQUIRE(SAME_SHAPE2D(reference_angle, streaming_angle));

    const RFLOAT tolerance = 1e-4;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(reference_score)
    {
        REQUIRE(DIRECT_MULTIDIM_ELEM(streaming_score, n) == Approx(DIRECT_MULTIDIM_ELEM(reference_score, n)).margin(tolerance));
        REQUIRE(DIRECT_MULTIDIM_ELEM(streaming_angle, n) == Approx(DIRECT_MULTIDIM_ELEM(reference_angle, n)).margin(tolerance));
    }
    REQUIRE(streaming_skew == Approx(reference_skew).margin(tolerance));
    REQUIRE(streaming_kurt == Approx(reference_kurt).margin(tolerance));
}
