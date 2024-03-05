function varargout = pde2(varargin)
% PDE2 MATLAB code for pde2.fig
%      PDE2, by itself, creates a new PDE2 or raises the existing
%      singleton*.
%
%      H = PDE2 returns the handle to a new PDE2 or the handle to
%      the existing singleton*.
%
%      PDE2('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PDE2.M with the given input arguments.
%
%      PDE2('Property','Value',...) creates a new PDE2 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before pde2_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to pde2_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help pde2

% Last Modified by GUIDE v2.5 16-Dec-2019 16:00:06


% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @pde2_OpeningFcn, ...
                   'gui_OutputFcn',  @pde2_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before pde2 is made visible.
function pde2_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to pde2 (see VARARGIN)

% Choose default command line output for pde2
handles.output = hObject;
handles.brain_model_has_be_chosen = 0;
handles.datasetok = 0;
handles.cort_sub_selected = 1;
set(handles.cortical_radio,'Value',1);
set(handles.subcortical_radio,'Value',0)
handles.lin_exp_selected = 1;
set(handles.linear_weight_func_radio,'Value',0);
set(handles.exp_weight_func_radio,'Value',0);
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes pde2 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = pde2_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in load_data.
function load_data_Callback(hObject, eventdata, handles)
% hObject    handle to load_data (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename1,filepath1]=uigetfile({'*.*','All Files'},...
    'Select Data File 1');
coordinates = dlmread([filepath1 filename1],';',1,0);
handles.datasetok = 0;
[rows cols] = size(coordinates)
if (cols == 3)
    msg = 'Weights not detected. Setting them to 1';
    title = 'Weights not detected';
    one = ones(rows,1);
    coordinates = [coordinates, one]
    f = warndlg(msg,title)
    uiwait(f) 
elseif ((cols > 4) || (cols < 3))
        msg = 'File format is not correct.'
        title = 'File not correct'
        f = errordlg(msg,title)
        return
end
handles.datasetok = 1;
handles.coordinates = coordinates;
guidata(hObject, handles);
clear_axes(handles)
show_datapoints(handles)

function show_datapoints(handles)
axes(handles.axes1);
scatter3(handles.coordinates(:,1), handles.coordinates(:,2), handles.coordinates(:,3), 50, 'filled')
hold on
axes(handles.axes2);
scatter3(handles.coordinates(:,1), handles.coordinates(:,2), handles.coordinates(:,3), 50, handles.coordinates(:,4), 'filled')
hold on
rotate3d on

function clear_axes(handles)
axes(handles.axes1);
cla()
axes(handles.axes2);
cla()


% --- Executes on button press in load_fs_average.
function load_fs_average_Callback(hObject, eventdata, handles)
% hObject    handle to load_fs_average (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
vertex_coordinates = load('FSAverage_327684_coordinates.mat');
handles.brain_vertex_coordinates = vertex_coordinates.coordinates_MNI;
G = load('FSAverage_327684_graph.mat', 'edges');
handles.G = graph(G.edges(:,1), G.edges(:,2), G.edges(:,3));
set(handles.brain_model_chosen,'string','FSAverage 327684');
handles.brain_model_has_be_chosen = 1;
guidata(hObject, handles);

% --- Executes on button press in load_ICBM152.
function load_ICBM152_Callback(hObject, eventdata, handles)
% hObject    handle to load_ICBM152 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
vertex_coordinates = load('ICBM152_coordinates.mat');
handles.brain_vertex_coordinates = vertex_coordinates.coordinates_MNI;
G = load('ICBM152_graph.mat', 'edges');
handles.G = graph(G.edges(:,1), G.edges(:,2), G.edges(:,3));
set(handles.brain_model_chosen,'string','ICBM152');
handles.brain_model_has_be_chosen = 1;
guidata(hObject, handles);



% --- Executes on button press in loadbraintemplate.
function loadbraintemplate_Callback(hObject, eventdata, handles)
% hObject    handle to loadbraintemplate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
selection = get(handles.brainmodelselector,'Value');
templates = get(handles.brainmodelselector,'String');
template_selected = templates{selection}
vertex_coordinates = load(strcat('./braintemplates/',template_selected,'_coordinates.mat'));
handles.brain_vertex_coordinates = vertex_coordinates.coordinates_MNI;
G = load(strcat('./braintemplates/',template_selected,'_graph.mat'), 'edges');
handles.G = graph(G.edges(:,1), G.edges(:,2), G.edges(:,3));
set(handles.brain_model_chosen,'string',template_selected);
handles.brain_model_has_be_chosen = 1;
guidata(hObject, handles);



function bandwidth_n_Callback(hObject, eventdata, handles)
% hObject    handle to bandwidth_n (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of bandwidth_n as text
%        str2double(get(hObject,'String')) returns contents of bandwidth_n as a double


% --- Executes during object creation, after setting all properties.
function bandwidth_n_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bandwidth_n (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function neighbours_n_Callback(hObject, eventdata, handles)
% hObject    handle to neighbours_n (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of neighbours_n as text
%        str2double(get(hObject,'String')) returns contents of neighbours_n as a double


% --- Executes during object creation, after setting all properties.
function neighbours_n_CreateFcn(hObject, eventdata, handles)
% hObject    handle to neighbours_n (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in estimate_pd.
function estimate_pd_Callback(hObject, eventdata, handles)
% hObject    handle to estimate_pd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if handles.datasetok == 0
     msg = 'Please, load dataset first.'
     title = 'Dataset error'
     f = errordlg(msg,title)
     return
end

if handles.cort_sub_selected == 0
     msg = 'Please, select cortical or subcortical analysis.'
     title = 'Model selection'
     f = errordlg(msg,title)
     return
end
% Deve essere caricato solo se è stato caricato il corticale
if handles.brain_model_has_be_chosen == 0
     msg = 'Please, select brain model first.'
     title = 'Brain model error'
     f = errordlg(msg,title)
     return
end

if (handles.cortical_radio.Value == 1)
    estimate_pde_cortical(hObject, handles)
elseif (handles.subcortical_radio.Value == 1)
        estimate_pde_subcortical(hObject, handles)
        
end
% 


% --- Executes on button press in save_nifti.
function save_nifti_Callback(hObject, eventdata, handles)
% hObject    handle to save_nifti (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename1,filepath1]=uigetfile({'*.nii','All Files'},...
    'Select a nifti template');
nifti = load_nii([filepath1, filename1]);
nifti.img(:,:,:) = 0;
nifti_w = nifti;
coordinates = handles.pde_points;
probabilities = handles.prob_to_save;
probabilities_w = handles.prob_to_save_w;
square = str2num(handles.vox_fill.String);
for kk= 1:length(coordinates)
   [xx yy zz] = mni2orFROMxyz(coordinates(kk,1),coordinates(kk,2),coordinates(kk,3),1,'mni');
   nifti.img(floor(xx), floor(yy), floor(zz)) = probabilities(kk);
   nifti.img(-square+floor(xx):square+floor(xx), -square+floor(yy):square+floor(yy), -square+floor(zz):square+floor(zz)) = probabilities(kk);
   nifti_w.img(floor(xx), floor(yy), floor(zz)) = probabilities_w(kk);
   nifti_w.img(-square+floor(xx):square+floor(xx), -square+floor(yy):square+floor(yy), -square+floor(zz):square+floor(zz)) = probabilities_w(kk);
end
filename = handles.filename.String{1};
niftisave = strcat(filename,'.nii')
niftisave_w = strcat(filename,'_w.nii');
save_nii(nifti,niftisave);
save_nii(nifti_w,niftisave_w);
niftisave(1)
%[filename1,filepath1]=uigetfile({'*.nii','All Files'},...
%    'Select Data File 1');
% = load_nii([filepath1, filename1]);
nii = load_nii(niftisave);
view_nii(nii);



function filename_Callback(hObject, eventdata, handles)
% hObject    handle to filename (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of filename as text
%        str2double(get(hObject,'String')) returns contents of filename as a double


% --- Executes during object creation, after setting all properties.
function filename_CreateFcn(hObject, eventdata, handles)
% hObject    handle to filename (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in estimate_weighted_pd.
function estimate_weighted_pd_Callback(hObject, eventdata, handles)
% hObject    handle to estimate_weighted_pd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in cortical_radio.
function cortical_radio_Callback(hObject, eventdata, handles)
% hObject    handle to cortical_radio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of cortical_radio
handles.cort_sub_selected = 1
set(handles.subcortical_radio,'Value',0)
guidata(hObject, handles);


% --- Executes on button press in subcortical_radio.
function subcortical_radio_Callback(hObject, eventdata, handles)
% hObject    handle to subcortical_radio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of subcortical_radio
handles.cort_sub_selected = 1
set(handles.cortical_radio,'Value',0)
guidata(hObject, handles);

function count = get_cell_linear_exp(handles, weight, multiplier)
if handles.linear_weight_func_radio.Value == 1
    count = weight*multiplier;
else
    count = exp((weight)*multiplier);
end


function estimate_pde_subcortical(hObject, handles)
stim_coords = handles.coordinates(:,1:3)
weights = handles.coordinates(:,4)
min_weight = min(weights);
max_weight = max(weights);
boundary_surface = boundary(stim_coords);
set(handles.figure1, 'pointer', 'arrow')
axes(handles.axes1);
cla
hold on
%trisurf(boundary_surface,stim_coords(:,1),stim_coords(:,2),stim_coords(:,3),'Facecolor','red','FaceAlpha',0.1);
id_points = unique(boundary_surface);
mean_val = mean(stim_coords);
scale = str2double(handles.volume_multiplier.String);
[len_stim_point column] = size(stim_coords);
scaled_stim_point = scale.*(stim_coords-mean_val)+mean_val;
boundary_scaled_stim_point = boundary(scaled_stim_point);
%trisurf(boundary_scaled_stim_point,scaled_stim_point(:,1),scaled_stim_point(:,2),scaled_stim_point(:,3),'Facecolor','blue','FaceAlpha',0.1);
max_dist=[0 0 0];
for i=1:3
    for j=1:len_stim_point
        curr_dist = abs(mean_val(i) - scaled_stim_point(j,i));
        if max_dist(i) < curr_dist
            max_dist(i) = curr_dist;
        end
    end
end
max_side = max(max_dist).*2;
xc=mean_val(1); 
yc=mean_val(2); 
zc=mean_val(3);    % coordinated of the center
alpha=0.3;           % transparency (max=1=opaque)

X = [0 0 0 0 0 1; 1 0 1 1 1 1; 1 0 1 1 1 1; 0 0 0 0 0 1];
Y = [0 0 0 0 1 0; 0 1 0 0 1 1; 0 1 1 1 1 1; 0 0 1 1 1 0];
Z = [0 0 1 0 0 0; 0 0 1 0 0 0; 1 1 1 0 1 1; 1 1 1 0 1 1];

C='blue';                  % unicolor

X = max_side*(X-0.5) + xc;
Y = max_side*(Y-0.5) + yc;
Z = max_side*(Z-0.5) + zc; 

%fill3(X,Y,Z,C,'FaceAlpha',alpha);    % draw cube
axis equal
sampling = str2num(handles.sampling.String);
[X,Y,Z] = meshgrid(linspace(xc-max_side/2,xc+max_side/2,sampling),linspace(yc-max_side/2,yc+max_side/2,sampling), linspace(zc-max_side/2,zc+max_side/2,sampling));
% scatter3(X(:),Y(:),Z(:),2)
points_list = zeros(sampling^3,3);
for i=1:sampling^3
    points_list(i,:) = [X(i), Y(i), Z(i)];
end
tri = delaunayn(scaled_stim_point);
tn = tsearchn(scaled_stim_point, tri, points_list); 
IsInside = ~isnan(tn);
index_inside = find(IsInside);
% scatter3(points_list(index_inside,1),points_list(index_inside,2),points_list(index_inside,3), 20, 'filled')
points_list = points_list(index_inside,:)
[n_vertex column] = size(points_list)
probabilities = zeros(1,n_vertex);
probabilities_w = zeros(1,n_vertex);
w_h = str2num(handles.bandwidth_n.String);
multiplier = str2double(handles.weigths_multiplier.String);
min_weight = min(handles.coordinates(:,4));
max_weight = max(handles.coordinates(:,4));
parfor ii=1:n_vertex
    if (mod(ii, 100) == 0)
        disp(['Vertex ',num2str(ii),'/',num2str(n_vertex)])
    end
    probability = 0;
    probability_w = 0;
    for jj=1:len_stim_point
        dist = pdist([stim_coords(jj,:);points_list(ii,:)],'euclidean');
        % phi = 1/((2*pi).^(2/2)*w_h.^2)*exp(-0.5*((dist/w_h).^2))
        phi = dist/w_h;
        if (phi <= 1/2)
            count = 1;
            if max_weight == min_weight
                count_w = 1;
            else
                count_w = get_cell_linear_exp(handles, weights(jj), multiplier)
            end
        else
            count = 0;
            count_w = 0;
        end
        probability_w = probability_w + count_w;
        probability = probability + count;
    end 
    probability = probability/(len_stim_point*(w_h.^2));
    probability_w = probability_w/(len_stim_point*(w_h.^2));
    probabilities(ii) = probability;
    probabilities_w(ii) = probability_w;
    
end
%set(handles.figure1, 'pointer', 'arrow')
%axes(handles.axes1);
hold on 
size(points_list);
non_zero = find(probabilities);
non_zero_w = find(probabilities_w);
set(handles.figure1, 'pointer', 'arrow')
axes(handles.axes1);
cla
h2 = scatter3(points_list(non_zero,1), points_list(non_zero,2), points_list(non_zero,3), 20, probabilities(non_zero), 'filled');
axes(handles.axes2);
cla
h3 = scatter3(points_list(non_zero_w,1), points_list(non_zero_w,2), points_list(non_zero_w,3), 20, probabilities_w(non_zero_w), 'filled');
handles.pde_points = [points_list(non_zero,1), points_list(non_zero,2), points_list(non_zero,3)];
handles.pde_points_w = [points_list(non_zero_w,1), points_list(non_zero_w,2), points_list(non_zero_w,3)];
handles.probabilities = probabilities(non_zero);
handles.probabilities_w = probabilities_w(non_zero_w);
handles.prob_to_save = probabilities;
handles.prob_to_save_w = probabilities_w;
guidata(hObject, handles);

function estimate_pde_cortical(hObject, handles)
b_vertex_coord = handles.brain_vertex_coordinates;
stim_coords = handles.coordinates;
stimulation_vertex_id = [];
for ii=1:length(stim_coords)
    dist = (b_vertex_coord(:,1) - stim_coords(ii,1)).^ 2 + (b_vertex_coord(:,2) - stim_coords(ii,2)).^ 2 + (b_vertex_coord(:,3) - stim_coords(ii,3)).^ 2;
    stimulation_vertex_id(end+1) = find(dist==min(dist));
end
nearest_length = str2num(handles.neighbours_n.String);
w_h = str2num(handles.bandwidth_n.String);
[null len_stim_point] = size(stimulation_vertex_id);
nearest_vertex = [];
G = handles.G;
for ii=1:len_stim_point
    this_near = nearest(G,stimulation_vertex_id(ii), nearest_length);
    [n_near, null] = size(this_near);
    for jj=1:n_near
        result = find(ismember(nearest_vertex,this_near(jj)));
        if isempty(result)
            nearest_vertex(end+1) = this_near(jj);
        end
    end
end
[null n_vertex] = size(nearest_vertex);
%Calcolo la probabilità.
probabilities = zeros(1,n_vertex);
probabilities_w = zeros(1,n_vertex);
'Calculating pde'

set(handles.figure1, 'pointer', 'watch')
drawnow;
weights = handles.coordinates(:,4);
min_weight = min(weights);
max_weight = max(weights);

multiplier = str2double(handles.weigths_multiplier.String);
parfor ii=1:n_vertex
    if (mod(ii, 100) == 0)
        disp(['Vertex ',num2str(ii),'/',num2str(n_vertex)])
    end
    probability = 0;
    probability_w = 0;
    for jj=1:len_stim_point
        [sp dist] = shortestpath(G,nearest_vertex(ii),stimulation_vertex_id(jj));
        % phi = 1/((2*pi).^(2/2)*w_h.^2)*exp(-0.5*((dist/w_h).^2))
        phi = dist/w_h;
        if (phi <= 1/2)
            count = 1;
            if max_weight == min_weight
                count_w = 1
            else
                count_w = get_cell_linear_exp(handles, weights(jj), multiplier);
            end
        else
            count = 0;
            count_w = 0;
        end
        probability_w = probability_w + count_w;
        probability = probability + count;
    end 
    probability = probability/(len_stim_point*(w_h.^2));
    probability_w = probability_w/(len_stim_point*(w_h.^2));
    probabilities(ii) = probability;
    probabilities_w(ii) = probability_w;
    
end
set(handles.figure1, 'pointer', 'arrow')
axes(handles.axes1);
cla
hold on 
h2 = scatter3(b_vertex_coord(nearest_vertex,1), b_vertex_coord(nearest_vertex,2), b_vertex_coord(nearest_vertex,3), 40, probabilities, 'filled');
axes(handles.axes2);
cla
h3 = scatter3(b_vertex_coord(nearest_vertex,1), b_vertex_coord(nearest_vertex,2), b_vertex_coord(nearest_vertex,3), 40, probabilities_w, 'filled');
handles.pde_points = [b_vertex_coord(nearest_vertex,1), b_vertex_coord(nearest_vertex,2), b_vertex_coord(nearest_vertex,3)];
handles.probabilities = probabilities;
handles.probabilities_w = probabilities_w;
handles.probabilites_coordinates = [b_vertex_coord(nearest_vertex,1), b_vertex_coord(nearest_vertex,2), b_vertex_coord(nearest_vertex,3)];
handles.prob_to_save = probabilities;
handles.prob_to_save_w = probabilities_w;
guidata(hObject, handles);



function weigths_multiplier_Callback(hObject, eventdata, handles)
% hObject    handle to weigths_multiplier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of weigths_multiplier as text
%        str2double(get(hObject,'String')) returns contents of weigths_multiplier as a double


% --- Executes during object creation, after setting all properties.
function weigths_multiplier_CreateFcn(hObject, eventdata, handles)
% hObject    handle to weigths_multiplier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function brain_model_chosen_Callback(hObject, eventdata, handles)
% hObject    handle to brain_model_chosen (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of brain_model_chosen as text
%        str2double(get(hObject,'String')) returns contents of brain_model_chosen as a double


% --- Executes during object creation, after setting all properties.
function brain_model_chosen_CreateFcn(hObject, eventdata, handles)
% hObject    handle to brain_model_chosen (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in resample_pde.
function resample_pde_Callback(hObject, eventdata, handles)
% hObject    handle to resample_pde (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
probabilities = handles.probabilities;
up_limit = str2num(handles.up_limit.String);
low_limit = str2num(handles.low_limit.String);
norm_data = (up_limit-low_limit)*(probabilities - min(probabilities)) / ( max(probabilities) - min(probabilities) )+low_limit;
handles.prob_to_save = norm_data;
probabilities_w = handles.probabilities_w;
norm_data_w =  (up_limit-low_limit)*(probabilities_w - min(probabilities_w)) / ( max(probabilities_w) - min(probabilities_w) )+low_limit;
handles.prob_to_save_w = norm_data_w;
guidata(hObject, handles);
 
 


% --- Executes on button press in reset_pde.
function reset_pde_Callback(hObject, eventdata, handles)
% hObject    handle to reset_pde (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.prob_to_save = handles.probabilities;
handles.prob_to_save_w = handles.probabilities_w;
 



function up_limit_Callback(hObject, eventdata, handles)
% hObject    handle to up_limit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of up_limit as text
%        str2double(get(hObject,'String')) returns contents of up_limit as a double


% --- Executes during object creation, after setting all properties.
function up_limit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to up_limit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function volume_multiplier_Callback(hObject, eventdata, handles)
% hObject    handle to volume_multiplier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of volume_multiplier as text
%        str2double(get(hObject,'String')) returns contents of volume_multiplier as a double


% --- Executes during object creation, after setting all properties.
function volume_multiplier_CreateFcn(hObject, eventdata, handles)
% hObject    handle to volume_multiplier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function sampling_Callback(hObject, eventdata, handles)
% hObject    handle to sampling (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of sampling as text
%        str2double(get(hObject,'String')) returns contents of sampling as a double


% --- Executes during object creation, after setting all properties.
function sampling_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sampling (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function low_limit_Callback(hObject, eventdata, handles)
% hObject    handle to low_limit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of low_limit as text
%        str2double(get(hObject,'String')) returns contents of low_limit as a double


% --- Executes during object creation, after setting all properties.
function low_limit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to low_limit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in linear_weight_func_radio.
function linear_weight_func_radio_Callback(hObject, eventdata, handles)
% hObject    handle to linear_weight_func_radio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of linear_weight_func_radio
handles.lin_exp_selected = 1;
set(handles.exp_weight_func_radio,'Value',0);
guidata(hObject, handles);


% --- Executes on button press in exp_weight_func_radio.
function exp_weight_func_radio_Callback(hObject, eventdata, handles)
% hObject    handle to exp_weight_func_radio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of exp_weight_func_radio
handles.lin_exp_selected = 1;
set(handles.linear_weight_func_radio,'Value',0);
guidata(hObject, handles);


% --- Executes on button press in show_sampling.
function show_sampling_Callback(hObject, eventdata, handles)
if handles.cortical_radio.Value == 1
    show_sampling_cortical(handles)
else
    show_sampling_subcortical(handles)
end

function show_sampling_cortical(handles)
b_vertex_coord = handles.brain_vertex_coordinates;
stim_coords = handles.coordinates;
stimulation_vertex_id = [];
for ii=1:length(stim_coords)
    dist = (b_vertex_coord(:,1) - stim_coords(ii,1)).^ 2 + (b_vertex_coord(:,2) - stim_coords(ii,2)).^ 2 + (b_vertex_coord(:,3) - stim_coords(ii,3)).^ 2;
    stimulation_vertex_id(end+1) = find(dist==min(dist));
end
nearest_length = str2num(handles.neighbours_n.String);
w_h = str2num(handles.bandwidth_n.String);
[null len_stim_point] = size(stimulation_vertex_id);
nearest_vertex = [];
G = handles.G;
for ii=1:len_stim_point
    this_near = nearest(G,stimulation_vertex_id(ii), nearest_length);
    [n_near, null] = size(this_near);
    for jj=1:n_near
        result = find(ismember(nearest_vertex,this_near(jj)));
        if isempty(result)
            nearest_vertex(end+1) = this_near(jj);
        end
    end
end
assignin('base','nearest_vertex',sort(nearest_vertex))
clear_axes(handles)
show_datapoints(handles)
axes(handles.axes1);
hold on
h2 = scatter3(b_vertex_coord(nearest_vertex,1), b_vertex_coord(nearest_vertex,2), b_vertex_coord(nearest_vertex,3), 20, 'filled');

function show_sampling_subcortical(handles)
clear_axes(handles)
show_datapoints(handles)
hold on
% hObject    handle to show_sampling (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
stim_coords = handles.coordinates(:,1:3)
weights = handles.coordinates(:,4)
boundary_surface = boundary(stim_coords);
set(handles.figure1, 'pointer', 'arrow')
axes(handles.axes1);
hold on
trisurf(boundary_surface,stim_coords(:,1),stim_coords(:,2),stim_coords(:,3),'Facecolor','red','FaceAlpha',0.1);
id_points = unique(boundary_surface);
mean_val = mean(stim_coords);
scale = str2double(handles.volume_multiplier.String)
[len_stim_point column] = size(stim_coords);
scaled_stim_point = scale.*(stim_coords-mean_val)+mean_val;
boundary_scaled_stim_point = boundary(scaled_stim_point);
trisurf(boundary_scaled_stim_point,scaled_stim_point(:,1),scaled_stim_point(:,2),scaled_stim_point(:,3),'Facecolor','blue','FaceAlpha',0.1);
max_dist=[0 0 0];
for i=1:3
    for j=1:len_stim_point
        curr_dist = abs(mean_val(i) - scaled_stim_point(j,i));
        if max_dist(i) < curr_dist
            max_dist(i) = curr_dist;
        end
    end
end
max_side = max(max_dist).*2;
xc=mean_val(1); 
yc=mean_val(2); 
zc=mean_val(3);    % coordinated of the center
alpha=0.3;           % transparency (max=1=opaque)

X = [0 0 0 0 0 1; 1 0 1 1 1 1; 1 0 1 1 1 1; 0 0 0 0 0 1];
Y = [0 0 0 0 1 0; 0 1 0 0 1 1; 0 1 1 1 1 1; 0 0 1 1 1 0];
Z = [0 0 1 0 0 0; 0 0 1 0 0 0; 1 1 1 0 1 1; 1 1 1 0 1 1];

C='blue';                  % unicolor

X = max_side*(X-0.5) + xc;
Y = max_side*(Y-0.5) + yc;
Z = max_side*(Z-0.5) + zc; 

%fill3(X,Y,Z,C,'FaceAlpha',alpha);    % draw cube
axis equal
sampling = str2num(handles.sampling.String);
[X,Y,Z] = meshgrid(linspace(xc-max_side/2,xc+max_side/2,sampling),linspace(yc-max_side/2,yc+max_side/2,sampling), linspace(zc-max_side/2,zc+max_side/2,sampling));
% scatter3(X(:),Y(:),Z(:),2)
points_list = zeros(sampling^3,3);
for i=1:sampling^3
    points_list(i,:) = [X(i), Y(i), Z(i)];
end
tri = delaunayn(scaled_stim_point);
tn = tsearchn(scaled_stim_point, tri, points_list); 
IsInside = ~isnan(tn);
index_inside = find(IsInside);
scatter3(points_list(index_inside,1),points_list(index_inside,2),points_list(index_inside,3), 20, 'filled')



function vox_fill_Callback(hObject, eventdata, handles)
% hObject    handle to vox_fill (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of vox_fill as text
%        str2double(get(hObject,'String')) returns contents of vox_fill as a double


% --- Executes during object creation, after setting all properties.
function vox_fill_CreateFcn(hObject, eventdata, handles)
% hObject    handle to vox_fill (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in export_stim_points.
function export_stim_points_Callback(hObject, eventdata, handles)
% hObject    handle to export_stim_points (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[filename1,filepath1]=uigetfile({'*.nii','All Files'},...
    'Select a nifti template');
nifti = load_nii([filepath1, filename1]);
nifti.img(:,:,:) = 0;
average = nifti.img;
nifti_w = nifti;
coordinates = handles.coordinates(:,1:3);
square = str2num(handles.vox_fill_stim_points.String);
value = get(handles.checkbox_weights, 'Value');
maxvalue = max(handles.coordinates(:, 4))
for kk= 1:length(coordinates)
   [xx yy zz] = mni2orFROMxyz(coordinates(kk,1),coordinates(kk,2),coordinates(kk,3),1,'mni');
   if(value ==1)
       to_save = handles.coordinates(kk, 4) * 100
       nifti.img(floor(xx), floor(yy), floor(zz)) = to_save +  nifti.img(floor(xx), floor(yy), floor(zz));
       average(floor(xx), floor(yy), floor(zz)) = average(floor(xx), floor(yy), floor(zz)) + 1;
       nifti.img(-square+floor(xx):square+floor(xx), -square+floor(yy):square+floor(yy), -square+floor(zz):square+floor(zz)) = to_save + nifti.img(-square+floor(xx):square+floor(xx), -square+floor(yy):square+floor(yy), -square+floor(zz):square+floor(zz));
       average(-square+floor(xx):square+floor(xx), -square+floor(yy):square+floor(yy), -square+floor(zz):square+floor(zz)) = 1 + average(-square+floor(xx):square+floor(xx), -square+floor(yy):square+floor(yy), -square+floor(zz):square+floor(zz));
   else
       nifti.img(floor(xx), floor(yy), floor(zz)) = 1;
       nifti.img(-square+floor(xx):square+floor(xx), -square+floor(yy):square+floor(yy), -square+floor(zz):square+floor(zz)) = 1;
   end
end
if(value ==1)
    average(average == 0) = 1;
    nifti.img = nifti.img./average;
end
filename = handles.stim_points_filename.String{1};
niftisave = strcat(filename,'.nii');
save_nii(nifti,niftisave);
%[filename1,filepath1]=uigetfile({'*.nii','All Files'},...
%    'Select Data File 1');
% = load_nii([filepath1, filename1]);
nii = load_nii(niftisave);



function stim_points_filename_Callback(hObject, eventdata, handles)
% hObject    handle to stim_points_filename (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of stim_points_filename as text
%        str2double(get(hObject,'String')) returns contents of stim_points_filename as a double


% --- Executes during object creation, after setting all properties.
function stim_points_filename_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stim_points_filename (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function vox_fill_stim_points_Callback(hObject, eventdata, handles)
% hObject    handle to vox_fill_stim_points (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of vox_fill_stim_points as text
%        str2double(get(hObject,'String')) returns contents of vox_fill_stim_points as a double


% --- Executes during object creation, after setting all properties.
function vox_fill_stim_points_CreateFcn(hObject, eventdata, handles)
% hObject    handle to vox_fill_stim_points (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in brainmodelselector.
function brainmodelselector_Callback(hObject, eventdata, handles)
% hObject    handle to brainmodelselector (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns brainmodelselector contents as cell array
%        contents{get(hObject,'Value')} returns selected item from brainmodelselector


% --- Executes during object creation, after setting all properties.
function brainmodelselector_CreateFcn(hObject, eventdata, handles)
% hObject    handle to brainmodelselector (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%listing = dir('*braintemplates*/*graph*');
brainmodels = dir('braintemplates/*graph*');
brainlist = {};
for ii = 1:length(brainmodels)
    brainlist{ii} = erase(brainmodels(ii).name, '_graph.mat');
end
set(hObject, 'String', brainlist);


% --- Executes on button press in checkbox_weights.
function checkbox_weights_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_weights (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_weights
