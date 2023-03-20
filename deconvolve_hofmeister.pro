
function pad, img, pad_left, pad_right, pad_top, pad_bottom, value = value
  if not keyword_set(value) then value = 0
  im_size = dim(img)
  new_size = im_size + [pad_left + pad_right, pad_top + pad_bottom]
  padded_img = dblarr(new_size)
  padded_img[*, *] = value
  padded_img[pad_left, pad_bottom] = img
  return, padded_img
end

pro pad_img_psf, img,  psf, large_psf = large_psf, value = value
  if not keyword_set(value) then value = 0
  if not keyword_set(large_psf) then large_psf = 0
  im_size =  dim(img)
    padsize_pad = 0.25*im_size
    padsize_large_psf = 0.5*im_size
  if large_psf then begin
    img = pad(img, padsize_large_psf[0], padsize_large_psf[0], padsize_large_psf[1], padsize_large_psf[1], value = value)
    img[where(img eq 0)] = 1e-15
  endif else begin
    psf = pad(psf, padsize_pad[0], padsize_pad[0], padsize_pad[1], padsize_pad[1], value = value)
    img = pad(img, padsize_pad[0], padsize_pad[0], padsize_pad[1], padsize_pad[1], value = value)
  endelse 
end


pro unpad_img_psf, img, psf, large_psf = large_psf
  if not keyword_set(large_psf) then large_psf = 0
  im_size =  dim(img)
  unpadsize_pad = 1/6. * im_size
  unpadsize_large_psf = 1/4. * im_size
  if large_psf then begin
    img = img[unpadsize_large_psf[0] : im_size[0] - unpadsize_large_psf[0] -1, unpadsize_large_psf[1] : im_size[1] - unpadsize_large_psf[1] -1]
  endif else begin
    img = img[unpadsize_pad[0] : im_size[0] - unpadsize_pad[0] -1, unpadsize_pad[1] : im_size[1] - unpadsize_pad[1] -1]
    psf = psf[unpadsize_pad[0] : im_size[0] - unpadsize_pad[0] -1, unpadsize_pad[1] : im_size[1] - unpadsize_pad[1] -1]
   endelse
end

pro get_boundary_boxes, mask, img, psf, bd_box_img = bd_box_img, bd_box_psf = bd_box_psf, bd_box_makeeven = bd_box_makeeven
  if ndim(mask) eq 1 then bd_box_img = mask
  if ndim(mask) eq 2 then begin
    mask = array_indices(mask, where(mask ne 0))
    bd_box_img = [ min(mask[0]), max(mask[0]), min(mask[1]), max(mask[1])]
  endif
  bd_box_makeeven = intarr(4)
  if ((bd_box_img[1] - bd_box_img[0]) mod 2) eq 0 then bd_box_makeeven[1] = 1
  if ((bd_box_img[3] - bd_box_img[2]) mod 2) eq 0 then bd_box_makeeven[3] = 1
  if (bd_box_img[1] + bd_box_makeeven[1]) eq (dim(img))[0] then begin
    bd_box_makeeven[0] -= 1
    bd_box_makeeven[1] -= 1
  endif
  if (bd_box_img[3] + bd_box_makeeven[3]) eq (dim(img))[1] then begin
    bd_box_makeeven[2] -= 1
    bd_box_makeeven[3] -= 1
  endif
  bd_box_img += bd_box_makeeven
  
  ;convert the boundary box of the mask to a corresponding boundary box for the psf
  bd_box_psf = [(dim(psf))[0]/2 - (bd_box_img[1] - bd_box_img[0] +1), (dim(psf))[0]/2 + (bd_box_img[1] - bd_box_img[0] +1) -1, $
                (dim(psf))[1]/2 - (bd_box_img[3] - bd_box_img[2] +1), (dim(psf))[1]/2 + (bd_box_img[3] - bd_box_img[2] +1) -1]
end


function estimate_scattered_light, img_in, psf_in, bd_box_img, large_psf = large_psf, pad = pad
  if not keyword_set(large_psf) then large_psf = 0
  if n_elements(pad) eq 0 then pad = 1  
  
  ;derive the scattered light into the boundary box region
  img = img_in
  psf = psf_in
  
  ;as we only want to derive the scattered light from the surrounding into the boundary box, set the image intensity in the boundary box to zero.
  ;as we only want to have the scattered light, we set the intrinsic intensity, i.e., the center of the psf, to zero.
  img[bd_box_img[0] : bd_box_img[1], bd_box_img[2] : bd_box_img[3]] = 0.
  psf[(dim(psf))[0]/2, (dim(psf))[1]/2] = 0.
  
  ;pad the image and psf to break the periodic boundary condition involved by the convolution in the fourier domain
  if pad then pad_img_psf, img, psf, large_psf = large_psf
 
  ;derive the fourier transform of the psf
  psf = shift(psf, (dim(psf))[0]/2, (dim(psf))[1]/2)
  psf = fft(psf, -1)
  ;derive the foureir transform of the image
  img = fft(img, -1)
  ;convolve it with the psf
  img_background = img * psf
  ;and transform it back to the spatial domain
  img_background = fft(img_background, 1)
  
  ;undo the padding
  if pad then unpad_img_psf, img_background, psf, large_psf = large_psf
end



function deconvolve_hofmeister, img_in, psf_in, iterations = iterations, mask = mask,  large_psf = large_psf, pad = pad, estimate_background = estimate_background, tolerance = tolerance
;  
;  Deconvolve an image with the point spread function
;
;  Perform image deconvolution on an image with the instrument
;  point spread function using the my algorithm published in ...
;
;  Parameters
;  ----------
;  img :2d array
;         An image.
;  psf : 2d array
;         The point spread function.
;  iterations: `int`
;         Maximum number of iterations
;  tolerance: 'float'
;         The image deconvolution stops when the maximum change from all pixels between the simulated observed image and the observed image is less than TOLERANCE counts.
;  mask: 1d array or 2d array
;         Allows to select an image subregion for which the convolution is done. By that, the algorithm can massively speeds up. Can be either a 1d array containing the four elements [left, bottom, right, top], or a 2d array masking the pixels that shall be deconvolved. Actually, if the 2d mask is used, from that the boundaries of the 1d array will be calculated, i.e., all pixels in a corresponding rectangualar box will be deconvolved.
;         At the moment, the dimensions of mask has to be smaller than half of the dimensions of the image. If you need to deconvolve a larger region, deconvolve the entire image instead.
;  estimate_background: 1/0
;         If a subregion deconvolution is used, it determines if an incoming scattered light estimate from the surrounding region to the subregion should be applied. This increases the fidelity of the result, but costs some computation time. Generally, it is required for average image intensities and below, but is not required for deconvolving bright image regions.
;  use_gpu: 1/0
;         If 1, the deconvolution will be performed on the GPU.
;  pad: 1/0
;         If true, increase the size of both the psf and the image by a factor of two, and pad the psf and image accordingly with zeros. As this is a fourier-based method, this breaks the symmetric boundary conditions involved in the fourier transform.
;  large_psf: 1/0
;         Usually, the PSF has the same dimension as the image, restricting scattered light to half of the image size. If set to true, the PSF given to the deconvolution has to be double the image size (that allows scattering over the full image range). The image will be padded with zeros to match the size of the full psf, and deconvolution is done over the full psf.
;
;  Returns
;  -------
;  2d array
;         The deconvolved image

  if not keyword_set(large_psf) then large_psf = 0
  if not keyword_set(mask) then mask = !NULL
  if not keyword_set(iterations) then iterations = 25
  if n_elements(pad) eq 0 then pad = 1
  if n_elements(estimate_background) eq 0 then estimate_background = 1
  if not keyword_set(tolerance) then tolerance = .1

  ;At the moment, the mask option only works if the shape of the selected region is smaller than 0.5 * the shape of the image  
  ;this factor determines the speed of convergence, and should be set between [0.1, 1.0]
  k = 1.
  
  ;create a a copy of the image and psf
  img = img_in
  psf = psf_in
  
  ;for a psf deconvolution, the length of the axis should be even. Thus, if the mask provieded is odd, make it even by adding one row and/or columng
  if mask ne !NULL then begin
    get_boundary_boxes, mask, img, psf, bd_box_img = bd_box_img, bd_box_psf = bd_box_psf, bd_box_makeeven = bd_box_makeeven
  
    if estimate_background then begin
      ;derive the scattered light of the surrounding into the boundary box, and correct the image for it.
      background = estimate_scattered_light(img, psf, bd_box_img, large_psf = large_psf, pad = pad)
      img = img - background
    endif
    ;cut the image and psf.
    img = img[bd_box_img[0] : bd_box_img[1], bd_box_img[2] : bd_box_img[3]]
    psf = psf[bd_box_psf[0] : bd_box_psf[1], bd_box_psf[2] : bd_box_psf[3]]
    large_psf = 1 ;If we cut the image, we definitively want to include scattered light over the entire subimage. Thus, large_psf is set to True. The psf boundary box has already been cut before accordingly.
  endif
  
  ;before the psf deconvolution, we have to pad the image and psf with zeros to break the periodic boundary conditions for the convolution in the fourier domain
  if pad then begin
    pad_img_psf, img, psf, large_psf = large_psf, value = !values.f_nan
    pad_mask = where(finite(img) eq 0)
    img[pad_mask] = 0
    psf[pad_mask] = 0
  endif

  img_decon = img
  tolerance = 0.1
  for n_iter = 0, iterations -1 do begin
    img_decon_last = img_decon
    img_decon_con = convol_fft(img_decon, psf, kernel_fft = kernel_fft, /no_padding)
    img_decon_con[pad_mask] = 0.
    ;derive how far we are off between the deconvolved imaged convolved with the psf and the observed image, i.e., how consistent we are
    deviations = img_decon_con - img
    ;and adjust the approximated deconvolved image accordingly
    img_decon -=  k * deviations
;    img_decon[where(img_decon lt 0)] = 0
    ;if the deconvolved image has converged, end the iterations
    dev =  max(abs(img_decon - img_decon_last))
    if dev lt tolerance then break
  endfor
  
  ;undo the padding
  if pad then unpad_img_psf, img_decon, psf, large_psf = large_psf
  ;if we had to enlarge the FOV of the mask to get an even pixel length of the axis, shrink the image again
  if mask ne !NULL then img_decon = img_decon[0 - bd_box_makeeven[0] : (dim(img_decon))[0] - bd_box_makeeven[1], $
                                              0 - bd_box_makeeven[2] : (dim(img_decon))[1] - bd_box_makeeven[3]]
  ;and we are done
  return, img_decon
end




