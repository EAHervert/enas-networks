function diff_matrix = diff_matrix(gt,input)
%Difference between gt image and some input
%   Takes a ground truth and noisy image (input) and returns the error
%   matrix diff_matrix which represents epsilon. This difference must first
%   cast the images to single precision so that the difference can be
%   better investigated.
gt_single = cast(gt, 'single');
input_single = cast(input, 'single');
diff_matrix = gt_single - input_single;
end