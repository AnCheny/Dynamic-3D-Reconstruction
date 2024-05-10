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

#include "E:\\vs_engineering\\structured_light\\structured_light\\include\\opencv2\\phase_unwrapping\phase_unwrapping.hpp"
#include "E:\\vs_engineering\\structured_light\\structured_light\\include\\opencv2\\phase_unwrapping\\histogramphaseunwrapping.hpp"

/** @defgroup phase_unwrapping Phase Unwrapping API

Two-dimensional phase unwrapping is found in different applications like terrain elevation estimation
in synthetic aperture radar (SAR), field mapping in magnetic resonance imaging or as a way of finding
corresponding pixels in structured light reconstruction with sinusoidal patterns.

Given a phase map, wrapped between [-pi; pi], phase unwrapping aims at finding the "true" phase map
by adding the right number of 2*pi to each pixel.

The problem is straightforward for perfect wrapped phase map, but real data are usually not noise-free.
Among the different algorithms that were developed, quality-guided phase unwrapping methods are fast
and efficient. They follow a path that unwraps high quality pixels first,
avoiding error propagation from the start.

In this module, a quality-guided phase unwrapping is implemented following the approach described in @cite histogramUnwrapping .

*/