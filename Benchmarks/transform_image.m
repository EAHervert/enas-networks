function transformed_image = transform_image(image, coeff)
%Transforms an image based on the log2 transform with coeff multiplier
%   Takes a grayscale image, applies the log2 transform, and then
%   multiplies the result by coeff. Returns 1 if value exceeds 1.
transformed_image = log2(1 + image) * coeff;
idx = transformed_image > 1;
transformed_image(idx) = 1;
end
