

# Possion-Blender



A personal implement of a Siggraph Paper **Poisson Image Editing**## UsageNow we have two pictures: A and B, and we need to cling B onto A with A as the background. 

**Picture A**

![](bg.jpg)

**Picture B**

![](fg.jpg)

* if we directly add B onto A, there exists obvious boundary, which seems not natural enough:

![](Naive.jpg)

* if we use Possion Blendering tools, after we solve the possion equation, we get a more natural blendering result:

![](Possion.jpg)

##Note

In my project, I solve the equation by using **Gaussian-Sadel** Iteration to acclerate. That means when we solve 

<img src="http://www.forkosh.com/mathtex.cgi? \Large AX=b">

we can construct an iteration as:

<img src="http://www.forkosh.com/mathtex.cgi? \Large X^{k+1}=B_G X^{k}+f_G">

where 

<img src="http://www.forkosh.com/mathtex.cgi? \Large B_G=(D-L)^{-1}U">

<img src="http://www.forkosh.com/mathtex.cgi? \Large f_G=(D-L)^{-1}b">

<img src="http://www.forkosh.com/mathtex.cgi? \Large A=D+L+U">

