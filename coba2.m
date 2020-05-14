function isofinish (p)
  set (gca, "PlotBoxAspectRatioMode", "manual", ...
            "PlotBoxAspectRatio", [1 1 1]);
  set (p, "FaceColor", "interp");
  ## set (p, "FaceLighting", "flat");
  ## light ("Position", [1 1 5]);  # Available with JHandles
endfunction

N = 15;    # Increase number of vertices in each direction
iso = .4;  # Change isovalue to .1 to display a sphere
lin = linspace (0, 2, N);
[x, y, z] = meshgrid (lin, lin, lin);
c = abs ((x-.5).^2 + (y-.5).^2 + (z-.5).^2);
figure (); # Open another figure window

subplot (2,2,1); view (-38, 20);
[f, v] = isosurface (x, y, z, c, iso);
p = patch ("Faces", f, "Vertices", v, "EdgeColor", "none");
cdat = rand (size (c));       # Compute random patch color data
isocolors (x, y, z, cdat, p); # Directly set colors of patch
isofinish (p);                # Call user function isofinish

subplot (2,2,2); view (-38, 20);
p = patch ("Faces", f, "Vertices", v, "EdgeColor", "none");
[r, g, b] = meshgrid (lin, 2-lin, 2-lin);
cdat = isocolors (x, y, z, c, v); # Compute color data vertices
set (p, "FaceVertexCData", cdat); # Set color data manually
isofinish (p);

subplot (2,2,3); view (-38, 20);
p = patch ("Faces", f, "Vertices", v, "EdgeColor", "none");
cdat = isocolors (r, g, b, c, p); # Compute color data patch
set (p, "FaceVertexCData", cdat); # Set color data manually
isofinish (p);

subplot (2,2,4); view (-38, 20);
p = patch ("Faces", f, "Vertices", v, "EdgeColor", "none");
r = g = b = repmat ([1:N] / N, [N, 1, N]); # Black to white
cdat = isocolors (x, y, z, r, g, b, v);
set (p, "FaceVertexCData", cdat);
isofinish (p);
