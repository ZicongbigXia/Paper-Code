
 %Code for the RNN in Example 3 of "Z. Xia, et al, Distributed Nonconvex Optimization Subject to Globally Coupled Constraints via Collaborative Neurodynamic Optimization" submmited to Neural Networks  
 

tend=10;%total time
tspan= [0 tend];   

%graph
s=[1:4 5];
t=[2:5 1];
G = graph(s,t);
A= adjacency(G);
L =laplacian(G);

%B-matrix
cc=eye(5);
B=diag(cc(1,:));
for k=2:5
B=blkdiag(B,diag(cc(k,:)));
end

%initial states
f0=zeros(105,1);
f0(1:5)=[0 5 3 -2 -2];
f0(6:10)=[9 5 -3 -2 2];
x0=[0 5 3 -2 -2]';
y0=[9 5 -3 -2 2]';
f0(31:55)=kron(ones(5,1),x0);
f0(56:80)=kron(ones(5,1),y0);
 
options=odeset('RelTol',1e-8,'AbsTol',1e-8);
[t,f]=ode15s(@(t,y) fun1(t,y,L,x0,y0,B,alpha),tspan,f0,options);
 
%2D circle and curves
% plot(f(:,1:5),f(:,6:10),'linewidth',1.5)  hold on
% xc = 0.8;
% yc = 2.2;
% r = 1;
% theta = linspace(0, 2*pi, 100);
% x = xc + r * cos(theta);
% y = yc + r * sin(theta);
% plot(x, y,'--','linewidth',1);
% axis equal; 
% plot(x0(1),y0(1),'-o','Markersize',4,'linewidth',1.5,'color','k','MarkerFaceColor','k');  
% plot(x0(2),y0(2),'-o','Markersize',4,'linewidth',1.5,'color','k','MarkerFaceColor','k');  
% plot(x0(3),y0(3),'-o','Markersize',4,'linewidth',1.5,'color','k','MarkerFaceColor','k');  
% plot(x0(4),y0(4),'-o','Markersize',4,'linewidth',1.5,'color','k','MarkerFaceColor','k');  
% plot(x0(5),y0(5),'-o','Markersize',4, 'linewidth',1.5,'color','k','MarkerFaceColor','k'); 
% %plot(xc,yc,'-x','Markersize',8, 'linewidth',1.5,'color','k','MarkerFaceColor','k'); 
% plot(f(end,1),f(end,6),'-*','Markersize',8, 'linewidth',1.5,'color','k','MarkerFaceColor','k');
% grid on

%other parameters
alpha=0.5;

%3D
theta = linspace(0, 2*pi, 100); 
z = linspace(0, max(t), 100);
[Theta, Z] = meshgrid(theta, z); 
X = 1 * cos(Theta) + 0.8; 
Y =1 * sin(Theta) + 2.2; 
surf(Z,X,Y,  'FaceAlpha', 0.3, 'EdgeColor', 'none'); hold on
plot3(t,f(:,1:5),f(:,6:10),'linewidth',1.5);  
plot3(t(1),x0(1),y0(1),'-o','Markersize',4,'linewidth',1.5,'color','k','MarkerFaceColor','k');  
plot3(t(1),x0(2),y0(2),'-o','Markersize',4,'linewidth',1.5,'color','k','MarkerFaceColor','k');  
plot3(t(1),x0(3),y0(3),'-o','Markersize',4,'linewidth',1.5,'color','k','MarkerFaceColor','k');  
plot3(t(1),x0(4),y0(4),'-o','Markersize',4,'linewidth',1.5,'color','k','MarkerFaceColor','k');  
plot3(t(1),x0(5),y0(5),'-o','Markersize',4, 'linewidth',1.5,'color','k','MarkerFaceColor','k'); 
        
function df=fun1(t,f,L,x0,y0,B,alpha)
x=f(1:5);
y=f(6:10);
lx=f(11:15);
ly=f(16:20);
l=f(21:25);
wl=f(26:30);
xx=f(31:55);
yy=f(56:80);
ll=f(81:105);
dfx=-x+x0;
dfy=-y+y0;


b=10000;
CC=kron(L,eye(5));
%\hatx and \hatlambda layers
Dxx=-b*sat(CC*xx+B*(xx-kron(ones(5,1),x)));
Dyy=-b*sat(CC*yy+B*(yy-kron(ones(5,1),y)));
Dll=-b*sat(CC*ll+B*(ll-kron(ones(5,1),l)));
hh=[ll(1:5)'*(-(xx(1:5)-0.8).*(xx(1:5)-0.8)-(yy(1:5)-2.2).*(yy(1:5)-2.2)+1);
    ll(6:10)'*(-(xx(6:10)-0.8).*(xx(6:10)-0.8)-(yy(6:10)-2.2).*(yy(6:10)-2.2)+1);
    ll(11:15)'*(-(xx(11:15)-0.8).*(xx(11:15)-0.8)-(yy(11:15)-2.2).*(yy(11:15)-2.2)+1);
    ll(16:20)'*(-(xx(16:20)-0.8).*(xx(16:20)-0.8)-(yy(16:20)-2.2).*(yy(16:20)-2.2)+1);
    ll(21:25)'*(-(xx(21:25)-0.8).*(xx(21:25)-0.8)-(yy(21:25)-2.2).*(yy(21:25)-2.2)+1)];
%x-layer
Dx=dfx-L*x-L*lx+l.*(x-0.8)+alpha*2*hh.*(2*l.*(x-0.8));
Dy=dfy-L*y-L*ly+l.*(y-2.2)+alpha*2*hh.*(2*l.*(y-2.2));
%w-layer
Dlx=L*x;
Dly=L*y;
%lambda-layer
Dl=-l+max(0,l-(x-0.8).*(x-0.8)-(y-2.2).*(y-2.2)+1-L*l-L*wl);
%\barlambda-layer
Dwl=L*l;
df=10*[Dx;Dy;Dlx;Dly;Dl;Dwl;Dxx;Dyy;Dll];
end

