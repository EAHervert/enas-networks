function grayscale_image = residual_to_grayscale(residual, method)
%Takes image residual and converts to grayscale image
%   Takes a color residual image and calculates the MSE/MAE of the three
%   components of a pixel to return the value.
[rows, cols, ~] = size(residual);

grayscale_image = zeros(rows, cols);
for i=1:rows
    for j=1:cols
        x = double(residual(i, j, 1));
        y = double(residual(i, j, 2));
        z = double(residual(i, j, 3));

        if strcmp(method, 'MSE')
            grayscale_image(i, j) = (x * x + y * y + z * z) / (3 * 255 ^ 2);
        elseif strcmp(method, 'MAE')
            grayscale_image(i, j) = (abs(x) + abs(y) + abs(z)) / (3 * 255);
        elseif strcmp(method, 'MAX')
            grayscale_image(i, j) = max([abs(x), abs(y), abs(z)]) / 255;
        end

    end
end
end
