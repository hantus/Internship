│Ћ$
Ў§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ"serve*2.0.02unknown8ню!
t
dense/kernelVarHandleOp*
shape
:##*
shared_namedense/kernel*
dtype0*
_output_shapes
: 
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:##
l

dense/biasVarHandleOp*
shape:#*
shared_name
dense/bias*
dtype0*
_output_shapes
: 
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:#
x
dense_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:##*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:##
p
dense_1/biasVarHandleOp*
shared_namedense_1/bias*
dtype0*
_output_shapes
: *
shape:#
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:#
x
dense_2/kernelVarHandleOp*
shape
:#*
shared_namedense_2/kernel*
dtype0*
_output_shapes
: 
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0*
_output_shapes

:#
p
dense_2/biasVarHandleOp*
_output_shapes
: *
shape:*
shared_namedense_2/bias*
dtype0
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes
:
f
	Adam/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
shared_nameAdam/beta_1*
dtype0*
_output_shapes
: *
shape: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
s
lstm/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	@ї*
shared_namelstm/kernel
l
lstm/kernel/Read/ReadVariableOpReadVariableOplstm/kernel*
dtype0*
_output_shapes
:	@ї
Є
lstm/recurrent_kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	#ї*&
shared_namelstm/recurrent_kernel
ђ
)lstm/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/recurrent_kernel*
dtype0*
_output_shapes
:	#ї
k
	lstm/biasVarHandleOp*
_output_shapes
: *
shape:ї*
shared_name	lstm/bias*
dtype0
d
lstm/bias/Read/ReadVariableOpReadVariableOp	lstm/bias*
dtype0*
_output_shapes	
:ї
w
lstm_1/kernelVarHandleOp*
shape:	#ї*
shared_namelstm_1/kernel*
dtype0*
_output_shapes
: 
p
!lstm_1/kernel/Read/ReadVariableOpReadVariableOplstm_1/kernel*
dtype0*
_output_shapes
:	#ї
І
lstm_1/recurrent_kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	#ї*(
shared_namelstm_1/recurrent_kernel
ё
+lstm_1/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm_1/recurrent_kernel*
dtype0*
_output_shapes
:	#ї
o
lstm_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:ї*
shared_namelstm_1/bias
h
lstm_1/bias/Read/ReadVariableOpReadVariableOplstm_1/bias*
dtype0*
_output_shapes	
:ї
^
totalVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
_output_shapes
: *
shape: *
shared_namecount*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
ѓ
Adam/dense/kernel/mVarHandleOp*
shape
:##*$
shared_nameAdam/dense/kernel/m*
dtype0*
_output_shapes
: 
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
dtype0*
_output_shapes

:##
z
Adam/dense/bias/mVarHandleOp*"
shared_nameAdam/dense/bias/m*
dtype0*
_output_shapes
: *
shape:#
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
dtype0*
_output_shapes
:#
є
Adam/dense_1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape
:##*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
dtype0*
_output_shapes

:##
~
Adam/dense_1/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:#*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:#*
dtype0
є
Adam/dense_2/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape
:#*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
dtype0*
_output_shapes

:#
~
Adam/dense_2/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
Ђ
Adam/lstm/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:	@ї*#
shared_nameAdam/lstm/kernel/m
z
&Adam/lstm/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/kernel/m*
dtype0*
_output_shapes
:	@ї
Ћ
Adam/lstm/recurrent_kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:	#ї*-
shared_nameAdam/lstm/recurrent_kernel/m
ј
0Adam/lstm/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/recurrent_kernel/m*
dtype0*
_output_shapes
:	#ї
y
Adam/lstm/bias/mVarHandleOp*!
shared_nameAdam/lstm/bias/m*
dtype0*
_output_shapes
: *
shape:ї
r
$Adam/lstm/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/bias/m*
dtype0*
_output_shapes	
:ї
Ё
Adam/lstm_1/kernel/mVarHandleOp*%
shared_nameAdam/lstm_1/kernel/m*
dtype0*
_output_shapes
: *
shape:	#ї
~
(Adam/lstm_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm_1/kernel/m*
dtype0*
_output_shapes
:	#ї
Ў
Adam/lstm_1/recurrent_kernel/mVarHandleOp*/
shared_name Adam/lstm_1/recurrent_kernel/m*
dtype0*
_output_shapes
: *
shape:	#ї
њ
2Adam/lstm_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm_1/recurrent_kernel/m*
dtype0*
_output_shapes
:	#ї
}
Adam/lstm_1/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:ї*#
shared_nameAdam/lstm_1/bias/m
v
&Adam/lstm_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_1/bias/m*
dtype0*
_output_shapes	
:ї
ѓ
Adam/dense/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:##*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
dtype0*
_output_shapes

:##
z
Adam/dense/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:#*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:#*
dtype0
є
Adam/dense_1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:##*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:##*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*$
shared_nameAdam/dense_1/bias/v*
dtype0*
_output_shapes
: *
shape:#
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
dtype0*
_output_shapes
:#
є
Adam/dense_2/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:#*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
dtype0*
_output_shapes

:#
~
Adam/dense_2/bias/vVarHandleOp*
shape:*$
shared_nameAdam/dense_2/bias/v*
dtype0*
_output_shapes
: 
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
dtype0*
_output_shapes
:
Ђ
Adam/lstm/kernel/vVarHandleOp*
_output_shapes
: *
shape:	@ї*#
shared_nameAdam/lstm/kernel/v*
dtype0
z
&Adam/lstm/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/kernel/v*
dtype0*
_output_shapes
:	@ї
Ћ
Adam/lstm/recurrent_kernel/vVarHandleOp*-
shared_nameAdam/lstm/recurrent_kernel/v*
dtype0*
_output_shapes
: *
shape:	#ї
ј
0Adam/lstm/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/recurrent_kernel/v*
dtype0*
_output_shapes
:	#ї
y
Adam/lstm/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:ї*!
shared_nameAdam/lstm/bias/v
r
$Adam/lstm/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/bias/v*
dtype0*
_output_shapes	
:ї
Ё
Adam/lstm_1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:	#ї*%
shared_nameAdam/lstm_1/kernel/v
~
(Adam/lstm_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm_1/kernel/v*
dtype0*
_output_shapes
:	#ї
Ў
Adam/lstm_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
shape:	#ї*/
shared_name Adam/lstm_1/recurrent_kernel/v*
dtype0
њ
2Adam/lstm_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm_1/recurrent_kernel/v*
dtype0*
_output_shapes
:	#ї
}
Adam/lstm_1/bias/vVarHandleOp*#
shared_nameAdam/lstm_1/bias/v*
dtype0*
_output_shapes
: *
shape:ї
v
&Adam/lstm_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_1/bias/v*
_output_shapes	
:ї*
dtype0

NoOpNoOp
эI
ConstConst"/device:CPU:0*▓I
valueеIBЦI BъI
ш
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
l
cell
 
state_spec
!regularization_losses
"	variables
#trainable_variables
$	keras_api
R
%regularization_losses
&	variables
'trainable_variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
R
/regularization_losses
0	variables
1trainable_variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
R
9regularization_losses
:	variables
;trainable_variables
<	keras_api
h

=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
░
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_rate)mќ*mЌ3mў4mЎ=mџ>mЏHmюImЮJmъKmЪLmаMmА)vб*vБ3vц4vЦ=vд>vДHvеIvЕJvфKvФLvгMvГ
 
V
H0
I1
J2
K3
L4
M5
)6
*7
38
49
=10
>11
V
H0
I1
J2
K3
L4
M5
)6
*7
38
49
=10
>11
џ
Nlayer_regularization_losses
regularization_losses
Onon_trainable_variables

Players
	variables
trainable_variables
Qmetrics
 
 
 
 
џ
Rlayer_regularization_losses
regularization_losses
Snon_trainable_variables

Tlayers
	variables
trainable_variables
Umetrics
~

Hkernel
Irecurrent_kernel
Jbias
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
 
 

H0
I1
J2

H0
I1
J2
џ
Zlayer_regularization_losses
regularization_losses
[non_trainable_variables

\layers
	variables
trainable_variables
]metrics
 
 
 
џ
^layer_regularization_losses
regularization_losses
_non_trainable_variables

`layers
	variables
trainable_variables
ametrics
~

Kkernel
Lrecurrent_kernel
Mbias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
 
 

K0
L1
M2

K0
L1
M2
џ
flayer_regularization_losses
!regularization_losses
gnon_trainable_variables

hlayers
"	variables
#trainable_variables
imetrics
 
 
 
џ
jlayer_regularization_losses
%regularization_losses
knon_trainable_variables

llayers
&	variables
'trainable_variables
mmetrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
џ
nlayer_regularization_losses
+regularization_losses
onon_trainable_variables

players
,	variables
-trainable_variables
qmetrics
 
 
 
џ
rlayer_regularization_losses
/regularization_losses
snon_trainable_variables

tlayers
0	variables
1trainable_variables
umetrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
џ
vlayer_regularization_losses
5regularization_losses
wnon_trainable_variables

xlayers
6	variables
7trainable_variables
ymetrics
 
 
 
џ
zlayer_regularization_losses
9regularization_losses
{non_trainable_variables

|layers
:	variables
;trainable_variables
}metrics
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
ю
~layer_regularization_losses
?regularization_losses
non_trainable_variables
ђlayers
@	variables
Atrainable_variables
Ђmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUElstm/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUElstm/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUE	lstm/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUElstm_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_1/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUElstm_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
?
0
1
2
3
4
5
6
	7

8

ѓ0
 
 
 
 
 

H0
I1
J2

H0
I1
J2
ъ
 Ѓlayer_regularization_losses
Vregularization_losses
ёnon_trainable_variables
Ёlayers
W	variables
Xtrainable_variables
єmetrics
 
 

0
 
 
 
 
 
 

K0
L1
M2

K0
L1
M2
ъ
 Єlayer_regularization_losses
bregularization_losses
ѕnon_trainable_variables
Ѕlayers
c	variables
dtrainable_variables
іmetrics
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


Іtotal

їcount
Ї
_fn_kwargs
јregularization_losses
Ј	variables
љtrainable_variables
Љ	keras_api
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

І0
ї1
 
А
 њlayer_regularization_losses
јregularization_losses
Њnon_trainable_variables
ћlayers
Ј	variables
љtrainable_variables
Ћmetrics
 

І0
ї1
 
 
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/lstm/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/lstm/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/lstm_1/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_1/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/lstm_1/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/lstm/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/lstm/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/lstm_1/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_1/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/lstm_1/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
Ё
serving_default_lstm_inputPlaceholder* 
shape:         
@*
dtype0*+
_output_shapes
:         
@
С
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_inputlstm/kernellstm/recurrent_kernel	lstm/biaslstm_1/kernellstm_1/recurrent_kernellstm_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*-
_gradient_op_typePartitionedCall-141067*-
f(R&
$__inference_signature_wrapper_138373*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
Х
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOplstm/kernel/Read/ReadVariableOp)lstm/recurrent_kernel/Read/ReadVariableOplstm/bias/Read/ReadVariableOp!lstm_1/kernel/Read/ReadVariableOp+lstm_1/recurrent_kernel/Read/ReadVariableOplstm_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp&Adam/lstm/kernel/m/Read/ReadVariableOp0Adam/lstm/recurrent_kernel/m/Read/ReadVariableOp$Adam/lstm/bias/m/Read/ReadVariableOp(Adam/lstm_1/kernel/m/Read/ReadVariableOp2Adam/lstm_1/recurrent_kernel/m/Read/ReadVariableOp&Adam/lstm_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp&Adam/lstm/kernel/v/Read/ReadVariableOp0Adam/lstm/recurrent_kernel/v/Read/ReadVariableOp$Adam/lstm/bias/v/Read/ReadVariableOp(Adam/lstm_1/kernel/v/Read/ReadVariableOp2Adam/lstm_1/recurrent_kernel/v/Read/ReadVariableOp&Adam/lstm_1/bias/v/Read/ReadVariableOpConst*-
_gradient_op_typePartitionedCall-141132*(
f#R!
__inference__traced_save_141131*
Tout
2**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *8
Tin1
/2-	
Н
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm/kernellstm/recurrent_kernel	lstm/biaslstm_1/kernellstm_1/recurrent_kernellstm_1/biastotalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/lstm/kernel/mAdam/lstm/recurrent_kernel/mAdam/lstm/bias/mAdam/lstm_1/kernel/mAdam/lstm_1/recurrent_kernel/mAdam/lstm_1/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/lstm/kernel/vAdam/lstm/recurrent_kernel/vAdam/lstm/bias/vAdam/lstm_1/kernel/vAdam/lstm_1/recurrent_kernel/vAdam/lstm_1/bias/v*-
_gradient_op_typePartitionedCall-141274*+
f&R$
"__inference__traced_restore_141273*
Tout
2**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *7
Tin0
.2,ВР
ЌO
Я
__inference__traced_save_141131
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop*
&savev2_lstm_kernel_read_readvariableop4
0savev2_lstm_recurrent_kernel_read_readvariableop(
$savev2_lstm_bias_read_readvariableop,
(savev2_lstm_1_kernel_read_readvariableop6
2savev2_lstm_1_recurrent_kernel_read_readvariableop*
&savev2_lstm_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop1
-savev2_adam_lstm_kernel_m_read_readvariableop;
7savev2_adam_lstm_recurrent_kernel_m_read_readvariableop/
+savev2_adam_lstm_bias_m_read_readvariableop3
/savev2_adam_lstm_1_kernel_m_read_readvariableop=
9savev2_adam_lstm_1_recurrent_kernel_m_read_readvariableop1
-savev2_adam_lstm_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop1
-savev2_adam_lstm_kernel_v_read_readvariableop;
7savev2_adam_lstm_recurrent_kernel_v_read_readvariableop/
+savev2_adam_lstm_bias_v_read_readvariableop3
/savev2_adam_lstm_1_kernel_v_read_readvariableop=
9savev2_adam_lstm_1_recurrent_kernel_v_read_readvariableop1
-savev2_adam_lstm_1_bias_v_read_readvariableop
savev2_1_const

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1ј
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_bdfbea4b88d044c78e92f6e7596f8f7f/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: І
SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:+*┤
valueфBД+B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE├
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0■
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop&savev2_lstm_kernel_read_readvariableop0savev2_lstm_recurrent_kernel_read_readvariableop$savev2_lstm_bias_read_readvariableop(savev2_lstm_1_kernel_read_readvariableop2savev2_lstm_1_recurrent_kernel_read_readvariableop&savev2_lstm_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop-savev2_adam_lstm_kernel_m_read_readvariableop7savev2_adam_lstm_recurrent_kernel_m_read_readvariableop+savev2_adam_lstm_bias_m_read_readvariableop/savev2_adam_lstm_1_kernel_m_read_readvariableop9savev2_adam_lstm_1_recurrent_kernel_m_read_readvariableop-savev2_adam_lstm_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop-savev2_adam_lstm_kernel_v_read_readvariableop7savev2_adam_lstm_recurrent_kernel_v_read_readvariableop+savev2_adam_lstm_bias_v_read_readvariableop/savev2_adam_lstm_1_kernel_v_read_readvariableop9savev2_adam_lstm_1_recurrent_kernel_v_read_readvariableop-savev2_adam_lstm_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: Ќ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ѕ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:├
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2╣
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
_output_shapes
:*
T0ќ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*т
_input_shapesМ
л: :##:#:##:#:#:: : : : : :	@ї:	#ї:ї:	#ї:	#ї:ї: : :##:#:##:#:#::	@ї:	#ї:ї:	#ї:	#ї:ї:##:#:##:#:#::	@ї:	#ї:ї:	#ї:	#ї:ї: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : 
В+
з
while_body_137724
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         #Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Е
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0Б
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:ї*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*'
_output_shapes
:         #*
T0_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         #*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         #*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         #*
T0Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_3AddV2while_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :         #:         #: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:  : : : : : : : : :	 :
 
╗
c
*__inference_dropout_1_layer_call_fn_140660

inputs
identityѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         #*-
_gradient_op_typePartitionedCall-138048*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_138037ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         #*
T0"
identityIdentity:output:0*&
_input_shapes
:         #22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ё
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_138044

inputs

identity_1N
IdentityIdentityinputs*'
_output_shapes
:         #*
T0[

Identity_1IdentityIdentity:output:0*'
_output_shapes
:         #*
T0"!

identity_1Identity_1:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
╠-
с
!sequential_lstm_while_body_135672&
"sequential_lstm_while_loop_counter,
(sequential_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!sequential_lstm_strided_slice_1_0a
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5#
sequential_lstm_strided_slice_1_
[tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:ъ
#TensorArrayV2Read/TensorListGetItemTensorListGetItem]tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         #*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: e
add_3AddV2"sequential_lstm_while_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: г

Identity_1Identity(sequential_lstm_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"D
sequential_lstm_strided_slice_1!sequential_lstm_strided_slice_1_0"!

identity_1Identity_1:output:0"╝
[tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor]tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : : :	 :
 :  : : : : 
В+
з
while_body_139422
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*'
_output_shapes
:         #*
T0_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         #*
T0T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         #*
T0V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
_output_shapes
: *
value	B :*
dtype0U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         #:         #: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:
 :  : : : : : : : : :	 
П
Њ
while_cond_137161
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::: : : : : :	 :
 :  : : : 
Ё
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_140708

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         #[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         #"!

identity_1Identity_1:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
Е
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_140756

inputs
identityѕQ
dropout/rateConst*
valueB
 *═╠L=*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0_
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*'
_output_shapes
:         #*
T0ї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0б
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*'
_output_shapes
:         #*
T0ћ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:         #R
dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: Ѕ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:         #a
dropout/mulMulinputsdropout/truediv:z:0*'
_output_shapes
:         #*
T0o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*'
_output_shapes
:         #*

SrcT0
i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*'
_output_shapes
:         #*
T0Y
IdentityIdentitydropout/mul_1:z:0*'
_output_shapes
:         #*
T0"
identityIdentity:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
Ф
╩
%__inference_lstm_layer_call_fn_139891

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_137590*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:         
#*
Tin
2*-
_gradient_op_typePartitionedCall-137602є
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         
#"
identityIdentity:output:0*6
_input_shapes%
#:         
@:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
Ё
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_140655

inputs

identity_1N
IdentityIdentityinputs*'
_output_shapes
:         #*
T0[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         #"!

identity_1Identity_1:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
х
┌
E__inference_lstm_cell_layer_call_and_return_conditional_losses_140822

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@їj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їp
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0U
mulMulSigmoid_1:y:0states_1*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         #*
T0T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         #*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #ю
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0ъ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*X
_input_shapesG
E:         @:         #:         #:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1
╦
b
C__inference_dropout_layer_call_and_return_conditional_losses_137633

inputs
identityѕQ
dropout/rateConst*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*+
_output_shapes
:         
#ї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0д
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*+
_output_shapes
:         
#ў
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*+
_output_shapes
:         
#R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0Ї
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*+
_output_shapes
:         
#*
T0e
dropout/mulMulinputsdropout/truediv:z:0*
T0*+
_output_shapes
:         
#s
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*+
_output_shapes
:         
#m
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         
#]
IdentityIdentitydropout/mul_1:z:0*
T0*+
_output_shapes
:         
#"
identityIdentity:output:0**
_input_shapes
:         
#:& "
 
_user_specified_nameinputs
н	
▄
C__inference_dense_2_layer_call_and_return_conditional_losses_138216

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:#i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         і
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         #::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ѕ
Џ
+__inference_sequential_layer_call_fn_138305

lstm_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityѕбStatefulPartitionedCall┐
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tin
2*'
_output_shapes
:         *-
_gradient_op_typePartitionedCall-138290*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_138289*
Tout
2**
config_proto

GPU 

CPU2J 8ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*Z
_input_shapesI
G:         
@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:* &
$
_user_specified_name
lstm_input: : : : : : : : :	 :
 : : 
╩'
Ї
F__inference_sequential_layer_call_and_return_conditional_losses_138261

lstm_input'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3)
%lstm_1_statefulpartitionedcall_args_1)
%lstm_1_statefulpartitionedcall_args_2)
%lstm_1_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбlstm/StatefulPartitionedCallбlstm_1/StatefulPartitionedCallЦ
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_input#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_137590*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:         
#*-
_gradient_op_typePartitionedCall-137602─
dropout/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-137652*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_137640*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:         
#┴
lstm_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0%lstm_1_statefulpartitionedcall_args_1%lstm_1_statefulpartitionedcall_args_2%lstm_1_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-138006*K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_137994*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2к
dropout_1/PartitionedCallPartitionedCall'lstm_1/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         #*-
_gradient_op_typePartitionedCall-138056*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_138044Ќ
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:         #*-
_gradient_op_typePartitionedCall-138078*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_138072*
Tout
2**
config_proto

GPU 

CPU2J 8┼
dropout_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         #*-
_gradient_op_typePartitionedCall-138128*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_138116*
Tout
2Ъ
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_138144*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         #*-
_gradient_op_typePartitionedCall-138150К
dropout_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-138200*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_138188*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         #Ъ
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         *
Tin
2*-
_gradient_op_typePartitionedCall-138222*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_138216ћ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*Z
_input_shapesI
G:         
@::::::::::::2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:* &
$
_user_specified_name
lstm_input: : : : : : : : :	 :
 : : 
П
Њ
while_cond_137021
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : ::::  : : : : : : : : :	 :
 
╠
╠
%__inference_lstm_layer_call_fn_139537
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3**
config_proto

GPU 

CPU2J 8*
Tin
2*4
_output_shapes"
 :                  #*-
_gradient_op_typePartitionedCall-136601*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_136600*
Tout
2Ј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*4
_output_shapes"
 :                  #*
T0"
identityIdentity:output:0*?
_input_shapes.
,:                  @:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0: : : 
и
F
*__inference_dropout_3_layer_call_fn_140771

inputs
identityЏ
PartitionedCallPartitionedCallinputs*'
_output_shapes
:         #*
Tin
2*-
_gradient_op_typePartitionedCall-138200*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_138188*
Tout
2**
config_proto

GPU 

CPU2J 8`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
Ч
Ќ
+__inference_sequential_layer_call_fn_139187

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityѕбStatefulPartitionedCall╗
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tin
2*'
_output_shapes
:         *-
_gradient_op_typePartitionedCall-138335*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_138334*
Tout
2**
config_proto

GPU 

CPU2J 8ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*Z
_input_shapesI
G:         
@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :	 :
 : : :& "
 
_user_specified_nameinputs: 
Г
п
E__inference_lstm_cell_layer_call_and_return_conditional_losses_136073

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@їj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Д
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їn
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         #*
T0T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         #*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #ю
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:         @:         #:         #:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates:&"
 
_user_specified_namestates: : : 
В+
з
while_body_137320
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Е
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         #*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         #*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         #:         #: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:  : : : : : : : : :	 :
 
нP
А
@__inference_lstm_layer_call_and_return_conditional_losses_137590

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         #O
zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         #c
transpose/permConst*
_output_shapes
:*!
valueB"          *
dtype0m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         @D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:         @*
T0*
Index0*
shrink_axis_maskБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ї|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0]
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         #*
T0T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         #*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
valueB"    #   *
dtype0Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
element_dtype0*
_output_shapes
: *

shape_type0F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_137489*L
_output_shapes:
8: : : : :         #:         #: : : : : *
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_137488*
_num_original_outputsK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:         #*
T0^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         #M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   ═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
         #h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         #*
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
#[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: «
IdentityIdentitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         
#"
identityIdentity:output:0*6
_input_shapes%
#:         
@:::2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: :& "
 
_user_specified_nameinputs: : 
и
▄
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_140916

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Д
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їp
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         #*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         #*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         #*
T0ю
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0ъ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0"!

identity_2Identity_2:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0*X
_input_shapesG
E:         #:         #:         #:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : : 
█P
Б
B__inference_lstm_1_layer_call_and_return_conditional_losses_140614

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
dtype0*
_output_shapes
: *
value
B :УY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
value	B :#*
dtype0s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
_output_shapes
:*
T0*
NP
zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:         #*
T0O
zeros_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :#c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         #c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         #D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         #Б
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#ї|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Д
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #]
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         #*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #n
TensorArrayV2_1/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_num_original_outputs*
bodyR
while_body_140513*L
_output_shapes:
8: : : : :         #:         #: : : : : *K
output_shapes:
8: : : : :         #:         #: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_140512K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:         #*
T0^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:         #*
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
         #h
strided_slice_3/stackConst*
_output_shapes
:*
valueB:
         *
dtype0a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         #*
Index0*
T0e
transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
#[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*6
_input_shapes%
#:         
#:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile: : :& "
 
_user_specified_nameinputs: 
┬
Х
lstm_1_while_cond_139027
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_1_strided_slice_12
.lstm_1_tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
W
LessLessplaceholderless_lstm_1_strided_slice_1*
_output_shapes
: *
T0k
Less_1Lesslstm_1_while_loop_counterlstm_1_while_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::: : : : : :	 :
 :  : : : 
»
┌
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_136680

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їn
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*'
_output_shapes
:         #*
T0_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #ю
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0ъ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*X
_input_shapesG
E:         #:         #:         #:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:&"
 
_user_specified_namestates: : : :& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
╠А
«
"__inference__traced_restore_141273
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate#
assignvariableop_11_lstm_kernel-
)assignvariableop_12_lstm_recurrent_kernel!
assignvariableop_13_lstm_bias%
!assignvariableop_14_lstm_1_kernel/
+assignvariableop_15_lstm_1_recurrent_kernel#
assignvariableop_16_lstm_1_bias
assignvariableop_17_total
assignvariableop_18_count+
'assignvariableop_19_adam_dense_kernel_m)
%assignvariableop_20_adam_dense_bias_m-
)assignvariableop_21_adam_dense_1_kernel_m+
'assignvariableop_22_adam_dense_1_bias_m-
)assignvariableop_23_adam_dense_2_kernel_m+
'assignvariableop_24_adam_dense_2_bias_m*
&assignvariableop_25_adam_lstm_kernel_m4
0assignvariableop_26_adam_lstm_recurrent_kernel_m(
$assignvariableop_27_adam_lstm_bias_m,
(assignvariableop_28_adam_lstm_1_kernel_m6
2assignvariableop_29_adam_lstm_1_recurrent_kernel_m*
&assignvariableop_30_adam_lstm_1_bias_m+
'assignvariableop_31_adam_dense_kernel_v)
%assignvariableop_32_adam_dense_bias_v-
)assignvariableop_33_adam_dense_1_kernel_v+
'assignvariableop_34_adam_dense_1_bias_v-
)assignvariableop_35_adam_dense_2_kernel_v+
'assignvariableop_36_adam_dense_2_bias_v*
&assignvariableop_37_adam_lstm_kernel_v4
0assignvariableop_38_adam_lstm_recurrent_kernel_v(
$assignvariableop_39_adam_lstm_bias_v,
(assignvariableop_40_adam_lstm_1_kernel_v6
2assignvariableop_41_adam_lstm_1_recurrent_kernel_v*
&assignvariableop_42_adam_lstm_1_bias_v
identity_44ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9б	RestoreV2бRestoreV2_1ј
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*┤
valueфBД+B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0к
RestoreV2/shape_and_slicesConst"/device:CPU:0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+Э
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*┬
_output_shapes»
г:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0y
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:}
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:Ђ
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:Ђ
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:|
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0*
dtype0	*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0~
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0~
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0}
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0ѕ
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:Ђ
AssignVariableOp_11AssignVariableOpassignvariableop_11_lstm_kernelIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:І
AssignVariableOp_12AssignVariableOp)assignvariableop_12_lstm_recurrent_kernelIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_lstm_biasIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:Ѓ
AssignVariableOp_14AssignVariableOp!assignvariableop_14_lstm_1_kernelIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:Ї
AssignVariableOp_15AssignVariableOp+assignvariableop_15_lstm_1_recurrent_kernelIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:Ђ
AssignVariableOp_16AssignVariableOpassignvariableop_16_lstm_1_biasIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0{
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:{
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0*
_output_shapes
 *
dtype0P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0Ѕ
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_kernel_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:Є
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_dense_bias_mIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:І
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_1_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype0P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0Ѕ
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_1_bias_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:І
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_2_kernel_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:Ѕ
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_2_bias_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:ѕ
AssignVariableOp_25AssignVariableOp&assignvariableop_25_adam_lstm_kernel_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:њ
AssignVariableOp_26AssignVariableOp0assignvariableop_26_adam_lstm_recurrent_kernel_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
_output_shapes
:*
T0є
AssignVariableOp_27AssignVariableOp$assignvariableop_27_adam_lstm_bias_mIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:і
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_lstm_1_kernel_mIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0ћ
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_lstm_1_recurrent_kernel_mIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:ѕ
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_lstm_1_bias_mIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:Ѕ
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_kernel_vIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:Є
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dense_bias_vIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
_output_shapes
:*
T0І
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_1_kernel_vIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:Ѕ
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_1_bias_vIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
_output_shapes
:*
T0І
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_2_kernel_vIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:Ѕ
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_2_bias_vIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:ѕ
AssignVariableOp_37AssignVariableOp&assignvariableop_37_adam_lstm_kernel_vIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:њ
AssignVariableOp_38AssignVariableOp0assignvariableop_38_adam_lstm_recurrent_kernel_vIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:є
AssignVariableOp_39AssignVariableOp$assignvariableop_39_adam_lstm_bias_vIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:і
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_lstm_1_kernel_vIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
_output_shapes
:*
T0ћ
AssignVariableOp_41AssignVariableOp2assignvariableop_41_adam_lstm_1_recurrent_kernel_vIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:ѕ
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_lstm_1_bias_vIdentity_42:output:0*
dtype0*
_output_shapes
 ї
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:х
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ђ
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0ј
Identity_44IdentityIdentity_43:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_44Identity_44:output:0*├
_input_shapes▒
«: :::::::::::::::::::::::::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_32AssignVariableOp_322$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2: : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 
ы
І
*__inference_lstm_cell_layer_call_fn_140869

inputs
states_0
states_1"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity

identity_1

identity_2ѕбStatefulPartitionedCall╠
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5**
config_proto

GPU 

CPU2J 8*
Tin

2*M
_output_shapes;
9:         #:         #:         #*-
_gradient_op_typePartitionedCall-136079*N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_136038*
Tout
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         #ё

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         #ё

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         #"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*X
_input_shapesG
E:         @:         #:         #:::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1
і
Њ
while_cond_139604
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::: : : : :	 :
 :  : : : : 
В+
з
while_body_139774
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0Б
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:ї*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0Z
mulMulSigmoid_1:y:0placeholder_3*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*'
_output_shapes
:         #*
T0_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         #*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         #*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         #*
T0Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         #:         #: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : : : : : : :	 :
 :  
В+
з
while_body_140513
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         #Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Е
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0Б
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0Z
mulMulSigmoid_1:y:0placeholder_3*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         #*
T0T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         #*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         #*
T0Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         #:         #: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : : : : :	 :
 :  
П
Њ
while_cond_136379
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::: : :	 :
 :  : : : : : : 
Е
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_138109

inputs
identityѕQ
dropout/rateConst*
valueB
 *═╠L=*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:         #ї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0б
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:         #ћ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:         #R
dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: Ѕ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:         #a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:         #o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*'
_output_shapes
:         #*

SrcT0
i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*'
_output_shapes
:         #*
T0Y
IdentityIdentitydropout/mul_1:z:0*'
_output_shapes
:         #*
T0"
identityIdentity:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
Ё
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_138188

inputs

identity_1N
IdentityIdentityinputs*'
_output_shapes
:         #*
T0[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         #"!

identity_1Identity_1:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
Ё
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_140761

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         #[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         #"!

identity_1Identity_1:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
¤	
▄
C__inference_dense_1_layer_call_and_return_conditional_losses_140729

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:##*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:         #*
T0а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:#v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         #P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         #І
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         #*
T0"
identityIdentity:output:0*.
_input_shapes
:         #::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
і
Њ
while_cond_137723
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : ::::  : : : : : : : : :	 :
 
П
Њ
while_cond_139993
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::: : : : : :	 :
 :  : : : 
╩B
№
B__inference_lstm_1_layer_call_and_return_conditional_losses_137102

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityѕбStatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
value	B :#*
dtype0s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         #O
zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:         #*
T0c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  #D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
valueB :
         *
dtype0А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         #*
Index0*
T0Ь
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*M
_output_shapes;
9:         #:         #:         #*
Tin

2*-
_gradient_op_typePartitionedCall-136721*P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_136680*
Tout
2**
config_proto

GPU 

CPU2J 8n
TensorArrayV2_1/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
element_dtype0*
_output_shapes
: *

shape_type0F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
_output_shapes
: *
valueB :
         *
dtype0T
while/loop_counterConst*
_output_shapes
: *
value	B : *
dtype0«
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5^StatefulPartitionedCall*
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_137021*
_num_original_outputs*
bodyR
while_body_137022*L
_output_shapes:
8: : : : :         #:         #: : : : : K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         #^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:         #*
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  #h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*'
_output_shapes
:         #*
T0*
Index0*
shrink_axis_maske
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :                  #*
T0[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: ѓ
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*'
_output_shapes
:         #*
T0"
identityIdentity:output:0*?
_input_shapes.
,:                  #:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile: : : :& "
 
_user_specified_nameinputs
њQ
Б
@__inference_lstm_layer_call_and_return_conditional_losses_139354
inputs_0"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskM
zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
_output_shapes
:*
T0*
NP
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         #O
zeros_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :#c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         #c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  @D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
_output_shapes
: *
valueB :
         *
dtype0А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         @Б
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ї|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
_output_shapes
: *
value	B :*
dtype0Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #]
mulMulSigmoid_1:y:0zeros_1:output:0*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         #*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #n
TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
         *
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_num_original_outputs*
bodyR
while_body_139255*L
_output_shapes:
8: : : : :         #:         #: : : : : *K
output_shapes:
8: : : : :         #:         #: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_139254K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         #^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         #M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  #h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         #e
transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :                  #*
T0[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: и
IdentityIdentitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :                  #"
identityIdentity:output:0*?
_input_shapes.
,:                  @:::2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:( $
"
_user_specified_name
inputs/0: : : 
Ј
a
C__inference_dropout_layer_call_and_return_conditional_losses_137640

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
#_

Identity_1IdentityIdentity:output:0*+
_output_shapes
:         
#*
T0"!

identity_1Identity_1:output:0**
_input_shapes
:         
#:& "
 
_user_specified_nameinputs
Й'
Ѕ
F__inference_sequential_layer_call_and_return_conditional_losses_138334

inputs'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3)
%lstm_1_statefulpartitionedcall_args_1)
%lstm_1_statefulpartitionedcall_args_2)
%lstm_1_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбlstm/StatefulPartitionedCallбlstm_1/StatefulPartitionedCallА
lstm/StatefulPartitionedCallStatefulPartitionedCallinputs#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-137602*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_137590*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:         
#*
Tin
2─
dropout/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-137652*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_137640*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:         
#*
Tin
2┴
lstm_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0%lstm_1_statefulpartitionedcall_args_1%lstm_1_statefulpartitionedcall_args_2%lstm_1_statefulpartitionedcall_args_3*K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_137994*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2*-
_gradient_op_typePartitionedCall-138006к
dropout_1/PartitionedCallPartitionedCall'lstm_1/StatefulPartitionedCall:output:0*
Tin
2*'
_output_shapes
:         #*-
_gradient_op_typePartitionedCall-138056*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_138044*
Tout
2**
config_proto

GPU 

CPU2J 8Ќ
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2*-
_gradient_op_typePartitionedCall-138078*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_138072*
Tout
2┼
dropout_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-138128*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_138116*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2Ъ
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2*-
_gradient_op_typePartitionedCall-138150*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_138144К
dropout_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2*-
_gradient_op_typePartitionedCall-138200*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_138188Ъ
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_138216*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         *
Tin
2*-
_gradient_op_typePartitionedCall-138222ћ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*Z
_input_shapesI
G:         
@::::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
нP
А
@__inference_lstm_layer_call_and_return_conditional_losses_137421

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskM
zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
dtype0*
_output_shapes
: *
value
B :УY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
_output_shapes
: *
T0P
zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:         #*
T0O
zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:         #*
T0c
transpose/permConst*
_output_shapes
:*!
valueB"          *
dtype0m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:
         @*
T0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
_output_shapes
: *
valueB :
         *
dtype0А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
element_dtype0*
_output_shapes
: *

shape_type0є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
Index0*
T0Б
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	@ї*
dtype0|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Д
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	#ї*
dtype0v
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #]
mulMulSigmoid_1:y:0zeros_1:output:0*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*'
_output_shapes
:         #*
T0_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         #*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #n
TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
_output_shapes
: *
value	B :
*
dtype0T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_137320*L
_output_shapes:
8: : : : :         #:         #: : : : : *
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_137319*
_num_original_outputsK
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         #^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         #M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
valueB"    #   *
dtype0═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
         #h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*'
_output_shapes
:         #*
Index0*
T0*
shrink_axis_maske
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
#[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: «
IdentityIdentitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         
#"
identityIdentity:output:0*6
_input_shapes%
#:         
@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs: : : 
Ё
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_138116

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         #[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         #"!

identity_1Identity_1:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
і
Њ
while_cond_137892
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::: : : : : : : :	 :
 :  : 
і
с
!sequential_lstm_while_cond_135671&
"sequential_lstm_while_loop_counter,
(sequential_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3(
$less_sequential_lstm_strided_slice_1;
7sequential_lstm_tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
`
LessLessplaceholder$less_sequential_lstm_strided_slice_1*
_output_shapes
: *
T0}
Less_1Less"sequential_lstm_while_loop_counter(sequential_lstm_while_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : ::::  : : : : : : : : :	 :
 
╬,
ц
lstm_1_while_body_139028
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_1_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_1_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Ћ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         #Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Е
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0Б
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: \
add_3AddV2lstm_1_while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Б

Identity_1Identitylstm_1_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0"ф
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"2
lstm_1_strided_slice_1lstm_1_strided_slice_1_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         #:         #: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : : : : : :	 :
 :  : 
Е
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_140650

inputs
identityѕQ
dropout/rateConst*
valueB
 *═╠L=*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0_
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:         #ї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0б
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:         #ћ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*'
_output_shapes
:         #*
T0R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0Ѕ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:         #a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:         #o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:         #i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*'
_output_shapes
:         #*
T0Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
»
┌
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_136715

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	#ї*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #ю
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0ъ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0ъ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*X
_input_shapesG
E:         #:         #:         #:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: :& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates:&"
 
_user_specified_namestates: : 
ш
Ї
,__inference_lstm_cell_1_layer_call_fn_140977

inputs
states_0
states_1"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity

identity_1

identity_2ѕбStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*M
_output_shapes;
9:         #:         #:         #*-
_gradient_op_typePartitionedCall-136738*P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_136715*
Tout
2**
config_proto

GPU 

CPU2J 8ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         #ё

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*'
_output_shapes
:         #*
T0ё

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         #"!

identity_2Identity_2:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0*X
_input_shapesG
E:         #:         #:         #:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : : 
Г
п
E__inference_lstm_cell_layer_call_and_return_conditional_losses_136038

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@їj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їn
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #U
mulMulSigmoid_1:y:0states_1*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*'
_output_shapes
:         #*
T0_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         #*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #ю
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0ъ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*X
_input_shapesG
E:         @:         #:         #:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates:&"
 
_user_specified_namestates: : : 
Х
╬
'__inference_lstm_1_layer_call_fn_140268
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-137103*K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_137102*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         #*
T0"
identityIdentity:output:0*?
_input_shapes.
,:                  #:::22
StatefulPartitionedCallStatefulPartitionedCall: : :( $
"
_user_specified_name
inputs/0: 
═	
┌
A__inference_dense_layer_call_and_return_conditional_losses_140676

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:##*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         #а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:#*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         #*
T0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         #І
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*.
_input_shapes
:         #::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
и
▄
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_140949

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їp
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         #*
T0ю
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:         #:         #:         #:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : : 
і
Њ
while_cond_139773
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : ::::  : : : : : : : : :	 :
 
В+
з
while_body_139605
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #Z
mulMulSigmoid_1:y:0placeholder_3*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         #*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         #:         #: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 :
 
Д
╠
'__inference_lstm_1_layer_call_fn_140630

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_137994*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         #*-
_gradient_op_typePartitionedCall-138006ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*6
_input_shapes%
#:         
#:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
¤	
▄
C__inference_dense_1_layer_call_and_return_conditional_losses_138144

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:##i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         #а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:#*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         #P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         #І
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*.
_input_shapes
:         #::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
╬,
ц
lstm_1_while_body_138624
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_1_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_1_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   Ћ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         #Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Е
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:ї*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*'
_output_shapes
:         #*
T0_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         #*
T0V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         #*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
_output_shapes
: *
value	B :*
dtype0N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
_output_shapes
: *
value	B :*
dtype0\
add_3AddV2lstm_1_while_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Б

Identity_1Identitylstm_1_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"!

identity_4Identity_4:output:0"2
lstm_1_strided_slice_1lstm_1_strided_slice_1_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"ф
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 :
 
Е
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_140703

inputs
identityѕQ
dropout/rateConst*
_output_shapes
: *
valueB
 *═╠L=*
dtype0C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:         #ї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0б
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:         #ћ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*'
_output_shapes
:         #*
T0R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: Ѕ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:         #a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:         #o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:         #i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         #Y
IdentityIdentitydropout/mul_1:z:0*'
_output_shapes
:         #*
T0"
identityIdentity:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
Ю
╝
while_body_137162
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0$
 statefulpartitionedcall_args_5_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5ѕбStatefulPartitionedCallѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         #ѓ
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0 statefulpartitionedcall_args_5_0*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin

2*M
_output_shapes;
9:         #:         #:         #*-
_gradient_op_typePartitionedCall-136738*P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_136715ц
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
element_dtype0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
value	B :*
dtype0J
addAddV2placeholderadd/y:output:0*
_output_shapes
: *
T0I
add_1/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_1AddV2while_loop_counteradd_1/y:output:0*
_output_shapes
: *
T0Z
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
_output_shapes
: *
T0k

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
_output_shapes
: *
T0Z

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: ё

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         #ё

Identity_5Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         #"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0"B
statefulpartitionedcall_args_5 statefulpartitionedcall_args_5_0*Q
_input_shapes@
>: : : : :         #:         #: : :::22
StatefulPartitionedCallStatefulPartitionedCall:  : : : : : : : : :	 :
 
Пэ
ђ

!__inference__wrapped_model_135963

lstm_input2
.sequential_lstm_matmul_readvariableop_resource4
0sequential_lstm_matmul_1_readvariableop_resource3
/sequential_lstm_biasadd_readvariableop_resource4
0sequential_lstm_1_matmul_readvariableop_resource6
2sequential_lstm_1_matmul_1_readvariableop_resource5
1sequential_lstm_1_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource5
1sequential_dense_2_matmul_readvariableop_resource6
2sequential_dense_2_biasadd_readvariableop_resource
identityѕб'sequential/dense/BiasAdd/ReadVariableOpб&sequential/dense/MatMul/ReadVariableOpб)sequential/dense_1/BiasAdd/ReadVariableOpб(sequential/dense_1/MatMul/ReadVariableOpб)sequential/dense_2/BiasAdd/ReadVariableOpб(sequential/dense_2/MatMul/ReadVariableOpб&sequential/lstm/BiasAdd/ReadVariableOpб%sequential/lstm/MatMul/ReadVariableOpб'sequential/lstm/MatMul_1/ReadVariableOpбsequential/lstm/whileб(sequential/lstm_1/BiasAdd/ReadVariableOpб'sequential/lstm_1/MatMul/ReadVariableOpб)sequential/lstm_1/MatMul_1/ReadVariableOpбsequential/lstm_1/whileO
sequential/lstm/ShapeShape
lstm_input*
T0*
_output_shapes
:m
#sequential/lstm/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: o
%sequential/lstm/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:o
%sequential/lstm/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:А
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask]
sequential/lstm/zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: Ј
sequential/lstm/zeros/mulMul&sequential/lstm/strided_slice:output:0$sequential/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: _
sequential/lstm/zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Ѕ
sequential/lstm/zeros/LessLesssequential/lstm/zeros/mul:z:0%sequential/lstm/zeros/Less/y:output:0*
_output_shapes
: *
T0`
sequential/lstm/zeros/packed/1Const*
dtype0*
_output_shapes
: *
value	B :#Б
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:`
sequential/lstm/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: ю
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:         #_
sequential/lstm/zeros_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :#Њ
sequential/lstm/zeros_1/mulMul&sequential/lstm/strided_slice:output:0&sequential/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: a
sequential/lstm/zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Ј
sequential/lstm/zeros_1/LessLesssequential/lstm/zeros_1/mul:z:0'sequential/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: b
 sequential/lstm/zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: Д
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0b
sequential/lstm/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: б
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:         #s
sequential/lstm/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:Љ
sequential/lstm/transpose	Transpose
lstm_input'sequential/lstm/transpose/perm:output:0*+
_output_shapes
:
         @*
T0d
sequential/lstm/Shape_1Shapesequential/lstm/transpose:y:0*
T0*
_output_shapes
:o
%sequential/lstm/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:q
'sequential/lstm/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:q
'sequential/lstm/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ф
sequential/lstm/strided_slice_1StridedSlice sequential/lstm/Shape_1:output:0.sequential/lstm/strided_slice_1/stack:output:00sequential/lstm/strided_slice_1/stack_1:output:00sequential/lstm/strided_slice_1/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskv
+sequential/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
valueB :
         *
dtype0Л
sequential/lstm/TensorArrayV2TensorListReserve4sequential/lstm/TensorArrayV2/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ќ
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:§
7sequential/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm/transpose:y:0Nsequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
element_dtype0*
_output_shapes
: *

shape_type0o
%sequential/lstm/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:q
'sequential/lstm/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:q
'sequential/lstm/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╣
sequential/lstm/strided_slice_2StridedSlicesequential/lstm/transpose:y:0.sequential/lstm/strided_slice_2/stack:output:00sequential/lstm/strided_slice_2/stack_1:output:00sequential/lstm/strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
T0*
Index0├
%sequential/lstm/MatMul/ReadVariableOpReadVariableOp.sequential_lstm_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@їг
sequential/lstm/MatMulMatMul(sequential/lstm/strided_slice_2:output:0-sequential/lstm/MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0К
'sequential/lstm/MatMul_1/ReadVariableOpReadVariableOp0sequential_lstm_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	#ї*
dtype0д
sequential/lstm/MatMul_1MatMulsequential/lstm/zeros:output:0/sequential/lstm/MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Ћ
sequential/lstm/addAddV2 sequential/lstm/MatMul:product:0"sequential/lstm/MatMul_1:product:0*(
_output_shapes
:         ї*
T0┴
&sequential/lstm/BiasAdd/ReadVariableOpReadVariableOp/sequential_lstm_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їъ
sequential/lstm/BiasAddBiasAddsequential/lstm/add:z:0.sequential/lstm/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0W
sequential/lstm/ConstConst*
_output_shapes
: *
value	B :*
dtype0a
sequential/lstm/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Т
sequential/lstm/splitSplit(sequential/lstm/split/split_dim:output:0 sequential/lstm/BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0t
sequential/lstm/SigmoidSigmoidsequential/lstm/split:output:0*'
_output_shapes
:         #*
T0v
sequential/lstm/Sigmoid_1Sigmoidsequential/lstm/split:output:1*'
_output_shapes
:         #*
T0Ї
sequential/lstm/mulMulsequential/lstm/Sigmoid_1:y:0 sequential/lstm/zeros_1:output:0*'
_output_shapes
:         #*
T0n
sequential/lstm/ReluRelusequential/lstm/split:output:2*
T0*'
_output_shapes
:         #Ј
sequential/lstm/mul_1Mulsequential/lstm/Sigmoid:y:0"sequential/lstm/Relu:activations:0*'
_output_shapes
:         #*
T0ё
sequential/lstm/add_1AddV2sequential/lstm/mul:z:0sequential/lstm/mul_1:z:0*
T0*'
_output_shapes
:         #v
sequential/lstm/Sigmoid_2Sigmoidsequential/lstm/split:output:3*
T0*'
_output_shapes
:         #k
sequential/lstm/Relu_1Relusequential/lstm/add_1:z:0*
T0*'
_output_shapes
:         #Њ
sequential/lstm/mul_2Mulsequential/lstm/Sigmoid_2:y:0$sequential/lstm/Relu_1:activations:0*
T0*'
_output_shapes
:         #~
-sequential/lstm/TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   Н
sequential/lstm/TensorArrayV2_1TensorListReserve6sequential/lstm/TensorArrayV2_1/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: V
sequential/lstm/timeConst*
dtype0*
_output_shapes
: *
value	B : j
(sequential/lstm/while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: d
"sequential/lstm/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Ы
sequential/lstm/whileWhile+sequential/lstm/while/loop_counter:output:01sequential/lstm/while/maximum_iterations:output:0sequential/lstm/time:output:0(sequential/lstm/TensorArrayV2_1:handle:0sequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0(sequential/lstm/strided_slice_1:output:0Gsequential/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0.sequential_lstm_matmul_readvariableop_resource0sequential_lstm_matmul_1_readvariableop_resource/sequential_lstm_biasadd_readvariableop_resource'^sequential/lstm/BiasAdd/ReadVariableOp&^sequential/lstm/MatMul/ReadVariableOp(^sequential/lstm/MatMul_1/ReadVariableOp*K
output_shapes:
8: : : : :         #:         #: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *-
cond%R#
!sequential_lstm_while_cond_135671*
_num_original_outputs*-
body%R#
!sequential_lstm_while_body_135672*L
_output_shapes:
8: : : : :         #:         #: : : : : k
sequential/lstm/while/IdentityIdentitysequential/lstm/while:output:0*
T0*
_output_shapes
: m
 sequential/lstm/while/Identity_1Identitysequential/lstm/while:output:1*
_output_shapes
: *
T0m
 sequential/lstm/while/Identity_2Identitysequential/lstm/while:output:2*
T0*
_output_shapes
: m
 sequential/lstm/while/Identity_3Identitysequential/lstm/while:output:3*
T0*
_output_shapes
: ~
 sequential/lstm/while/Identity_4Identitysequential/lstm/while:output:4*
T0*'
_output_shapes
:         #~
 sequential/lstm/while/Identity_5Identitysequential/lstm/while:output:5*
T0*'
_output_shapes
:         #m
 sequential/lstm/while/Identity_6Identitysequential/lstm/while:output:6*
_output_shapes
: *
T0m
 sequential/lstm/while/Identity_7Identitysequential/lstm/while:output:7*
_output_shapes
: *
T0m
 sequential/lstm/while/Identity_8Identitysequential/lstm/while:output:8*
T0*
_output_shapes
: m
 sequential/lstm/while/Identity_9Identitysequential/lstm/while:output:9*
T0*
_output_shapes
: o
!sequential/lstm/while/Identity_10Identitysequential/lstm/while:output:10*
_output_shapes
: *
T0Љ
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:§
2sequential/lstm/TensorArrayV2Stack/TensorListStackTensorListStack)sequential/lstm/while/Identity_3:output:0Isequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
         #x
%sequential/lstm/strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:q
'sequential/lstm/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:q
'sequential/lstm/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:О
sequential/lstm/strided_slice_3StridedSlice;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0.sequential/lstm/strided_slice_3/stack:output:00sequential/lstm/strided_slice_3/stack_1:output:00sequential/lstm/strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         #*
Index0*
T0u
 sequential/lstm/transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0к
sequential/lstm/transpose_1	Transpose;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)sequential/lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:         
#k
sequential/lstm/runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: ~
sequential/dropout/IdentityIdentitysequential/lstm/transpose_1:y:0*+
_output_shapes
:         
#*
T0k
sequential/lstm_1/ShapeShape$sequential/dropout/Identity:output:0*
_output_shapes
:*
T0o
%sequential/lstm_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:q
'sequential/lstm_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:q
'sequential/lstm_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ф
sequential/lstm_1/strided_sliceStridedSlice sequential/lstm_1/Shape:output:0.sequential/lstm_1/strided_slice/stack:output:00sequential/lstm_1/strided_slice/stack_1:output:00sequential/lstm_1/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0_
sequential/lstm_1/zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: Ћ
sequential/lstm_1/zeros/mulMul(sequential/lstm_1/strided_slice:output:0&sequential/lstm_1/zeros/mul/y:output:0*
_output_shapes
: *
T0a
sequential/lstm_1/zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Ј
sequential/lstm_1/zeros/LessLesssequential/lstm_1/zeros/mul:z:0'sequential/lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: b
 sequential/lstm_1/zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: Е
sequential/lstm_1/zeros/packedPack(sequential/lstm_1/strided_slice:output:0)sequential/lstm_1/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:b
sequential/lstm_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: б
sequential/lstm_1/zerosFill'sequential/lstm_1/zeros/packed:output:0&sequential/lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:         #a
sequential/lstm_1/zeros_1/mul/yConst*
_output_shapes
: *
value	B :#*
dtype0Ў
sequential/lstm_1/zeros_1/mulMul(sequential/lstm_1/strided_slice:output:0(sequential/lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: c
 sequential/lstm_1/zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Ћ
sequential/lstm_1/zeros_1/LessLess!sequential/lstm_1/zeros_1/mul:z:0)sequential/lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: d
"sequential/lstm_1/zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: Г
 sequential/lstm_1/zeros_1/packedPack(sequential/lstm_1/strided_slice:output:0+sequential/lstm_1/zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:d
sequential/lstm_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: е
sequential/lstm_1/zeros_1Fill)sequential/lstm_1/zeros_1/packed:output:0(sequential/lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:         #u
 sequential/lstm_1/transpose/permConst*
_output_shapes
:*!
valueB"          *
dtype0»
sequential/lstm_1/transpose	Transpose$sequential/dropout/Identity:output:0)sequential/lstm_1/transpose/perm:output:0*
T0*+
_output_shapes
:
         #h
sequential/lstm_1/Shape_1Shapesequential/lstm_1/transpose:y:0*
_output_shapes
:*
T0q
'sequential/lstm_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:s
)sequential/lstm_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:s
)sequential/lstm_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:х
!sequential/lstm_1/strided_slice_1StridedSlice"sequential/lstm_1/Shape_1:output:00sequential/lstm_1/strided_slice_1/stack:output:02sequential/lstm_1/strided_slice_1/stack_1:output:02sequential/lstm_1/strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0x
-sequential/lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
valueB :
         *
dtype0О
sequential/lstm_1/TensorArrayV2TensorListReserve6sequential/lstm_1/TensorArrayV2/element_shape:output:0*sequential/lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ў
Gsequential/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Ѓ
9sequential/lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm_1/transpose:y:0Psequential/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: q
'sequential/lstm_1/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:s
)sequential/lstm_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:s
)sequential/lstm_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:├
!sequential/lstm_1/strided_slice_2StridedSlicesequential/lstm_1/transpose:y:00sequential/lstm_1/strided_slice_2/stack:output:02sequential/lstm_1/strided_slice_2/stack_1:output:02sequential/lstm_1/strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         #*
T0*
Index0К
'sequential/lstm_1/MatMul/ReadVariableOpReadVariableOp0sequential_lstm_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#ї▓
sequential/lstm_1/MatMulMatMul*sequential/lstm_1/strided_slice_2:output:0/sequential/lstm_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ї╦
)sequential/lstm_1/MatMul_1/ReadVariableOpReadVariableOp2sequential_lstm_1_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їг
sequential/lstm_1/MatMul_1MatMul sequential/lstm_1/zeros:output:01sequential/lstm_1/MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Џ
sequential/lstm_1/addAddV2"sequential/lstm_1/MatMul:product:0$sequential/lstm_1/MatMul_1:product:0*(
_output_shapes
:         ї*
T0┼
(sequential/lstm_1/BiasAdd/ReadVariableOpReadVariableOp1sequential_lstm_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їц
sequential/lstm_1/BiasAddBiasAddsequential/lstm_1/add:z:00sequential/lstm_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їY
sequential/lstm_1/ConstConst*
_output_shapes
: *
value	B :*
dtype0c
!sequential/lstm_1/split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0В
sequential/lstm_1/splitSplit*sequential/lstm_1/split/split_dim:output:0"sequential/lstm_1/BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0x
sequential/lstm_1/SigmoidSigmoid sequential/lstm_1/split:output:0*'
_output_shapes
:         #*
T0z
sequential/lstm_1/Sigmoid_1Sigmoid sequential/lstm_1/split:output:1*
T0*'
_output_shapes
:         #Њ
sequential/lstm_1/mulMulsequential/lstm_1/Sigmoid_1:y:0"sequential/lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:         #r
sequential/lstm_1/ReluRelu sequential/lstm_1/split:output:2*
T0*'
_output_shapes
:         #Ћ
sequential/lstm_1/mul_1Mulsequential/lstm_1/Sigmoid:y:0$sequential/lstm_1/Relu:activations:0*
T0*'
_output_shapes
:         #і
sequential/lstm_1/add_1AddV2sequential/lstm_1/mul:z:0sequential/lstm_1/mul_1:z:0*'
_output_shapes
:         #*
T0z
sequential/lstm_1/Sigmoid_2Sigmoid sequential/lstm_1/split:output:3*'
_output_shapes
:         #*
T0o
sequential/lstm_1/Relu_1Relusequential/lstm_1/add_1:z:0*
T0*'
_output_shapes
:         #Ў
sequential/lstm_1/mul_2Mulsequential/lstm_1/Sigmoid_2:y:0&sequential/lstm_1/Relu_1:activations:0*
T0*'
_output_shapes
:         #ђ
/sequential/lstm_1/TensorArrayV2_1/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:█
!sequential/lstm_1/TensorArrayV2_1TensorListReserve8sequential/lstm_1/TensorArrayV2_1/element_shape:output:0*sequential/lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: X
sequential/lstm_1/timeConst*
value	B : *
dtype0*
_output_shapes
: l
*sequential/lstm_1/while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: f
$sequential/lstm_1/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ћ
sequential/lstm_1/whileWhile-sequential/lstm_1/while/loop_counter:output:03sequential/lstm_1/while/maximum_iterations:output:0sequential/lstm_1/time:output:0*sequential/lstm_1/TensorArrayV2_1:handle:0 sequential/lstm_1/zeros:output:0"sequential/lstm_1/zeros_1:output:0*sequential/lstm_1/strided_slice_1:output:0Isequential/lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:00sequential_lstm_1_matmul_readvariableop_resource2sequential_lstm_1_matmul_1_readvariableop_resource1sequential_lstm_1_biasadd_readvariableop_resource)^sequential/lstm_1/BiasAdd/ReadVariableOp(^sequential/lstm_1/MatMul/ReadVariableOp*^sequential/lstm_1/MatMul_1/ReadVariableOp*
parallel_iterations */
cond'R%
#sequential_lstm_1_while_cond_135837*
_num_original_outputs*/
body'R%
#sequential_lstm_1_while_body_135838*L
_output_shapes:
8: : : : :         #:         #: : : : : *K
output_shapes:
8: : : : :         #:         #: : : : : *
T
2*
_lower_using_switch_merge(o
 sequential/lstm_1/while/IdentityIdentity sequential/lstm_1/while:output:0*
T0*
_output_shapes
: q
"sequential/lstm_1/while/Identity_1Identity sequential/lstm_1/while:output:1*
_output_shapes
: *
T0q
"sequential/lstm_1/while/Identity_2Identity sequential/lstm_1/while:output:2*
T0*
_output_shapes
: q
"sequential/lstm_1/while/Identity_3Identity sequential/lstm_1/while:output:3*
T0*
_output_shapes
: ѓ
"sequential/lstm_1/while/Identity_4Identity sequential/lstm_1/while:output:4*'
_output_shapes
:         #*
T0ѓ
"sequential/lstm_1/while/Identity_5Identity sequential/lstm_1/while:output:5*'
_output_shapes
:         #*
T0q
"sequential/lstm_1/while/Identity_6Identity sequential/lstm_1/while:output:6*
_output_shapes
: *
T0q
"sequential/lstm_1/while/Identity_7Identity sequential/lstm_1/while:output:7*
T0*
_output_shapes
: q
"sequential/lstm_1/while/Identity_8Identity sequential/lstm_1/while:output:8*
T0*
_output_shapes
: q
"sequential/lstm_1/while/Identity_9Identity sequential/lstm_1/while:output:9*
_output_shapes
: *
T0s
#sequential/lstm_1/while/Identity_10Identity!sequential/lstm_1/while:output:10*
T0*
_output_shapes
: Њ
Bsequential/lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Ѓ
4sequential/lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack+sequential/lstm_1/while/Identity_3:output:0Ksequential/lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
         #z
'sequential/lstm_1/strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:s
)sequential/lstm_1/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:s
)sequential/lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
valueB:*
dtype0р
!sequential/lstm_1/strided_slice_3StridedSlice=sequential/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:00sequential/lstm_1/strided_slice_3/stack:output:02sequential/lstm_1/strided_slice_3/stack_1:output:02sequential/lstm_1/strided_slice_3/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         #w
"sequential/lstm_1/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:╠
sequential/lstm_1/transpose_1	Transpose=sequential/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0+sequential/lstm_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:         
#m
sequential/lstm_1/runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: Є
sequential/dropout_1/IdentityIdentity*sequential/lstm_1/strided_slice_3:output:0*
T0*'
_output_shapes
:         #─
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:##Ф
sequential/dense/MatMulMatMul&sequential/dropout_1/Identity:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         #┬
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:#Е
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         #*
T0r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         #ђ
sequential/dropout_2/IdentityIdentity#sequential/dense/Relu:activations:0*'
_output_shapes
:         #*
T0╚
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:##»
sequential/dense_1/MatMulMatMul&sequential/dropout_2/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         #к
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:#»
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         #v
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         #ѓ
sequential/dropout_3/IdentityIdentity%sequential/dense_1/Relu:activations:0*'
_output_shapes
:         #*
T0╚
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:#»
sequential/dense_2/MatMulMatMul&sequential/dropout_3/Identity:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0к
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:»
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0|
sequential/dense_2/SoftmaxSoftmax#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         Џ
IdentityIdentity$sequential/dense_2/Softmax:softmax:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp'^sequential/lstm/BiasAdd/ReadVariableOp&^sequential/lstm/MatMul/ReadVariableOp(^sequential/lstm/MatMul_1/ReadVariableOp^sequential/lstm/while)^sequential/lstm_1/BiasAdd/ReadVariableOp(^sequential/lstm_1/MatMul/ReadVariableOp*^sequential/lstm_1/MatMul_1/ReadVariableOp^sequential/lstm_1/while*'
_output_shapes
:         *
T0"
identityIdentity:output:0*Z
_input_shapesI
G:         
@::::::::::::2V
)sequential/lstm_1/MatMul_1/ReadVariableOp)sequential/lstm_1/MatMul_1/ReadVariableOp2R
'sequential/lstm/MatMul_1/ReadVariableOp'sequential/lstm/MatMul_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2R
'sequential/lstm_1/MatMul/ReadVariableOp'sequential/lstm_1/MatMul/ReadVariableOp2P
&sequential/lstm/BiasAdd/ReadVariableOp&sequential/lstm/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2N
%sequential/lstm/MatMul/ReadVariableOp%sequential/lstm/MatMul/ReadVariableOp22
sequential/lstm_1/whilesequential/lstm_1/while2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2.
sequential/lstm/whilesequential/lstm/while2T
(sequential/lstm_1/BiasAdd/ReadVariableOp(sequential/lstm_1/BiasAdd/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp:* &
$
_user_specified_name
lstm_input: : : : : : : : :	 :
 : : 
▓,
ќ
lstm_while_body_138443
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_strided_slice_1_0V
Rtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_strided_slice_1T
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   Њ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Е
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	#ї*
dtype0u
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
_output_shapes
: *
value	B :*
dtype0Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #Z
mulMulSigmoid_1:y:0placeholder_3*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
dtype0*
_output_shapes
: *
value	B :N
add_2AddV2placeholderadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: Z
add_3AddV2lstm_while_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: А

Identity_1Identitylstm_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #".
lstm_strided_slice_1lstm_strided_slice_1_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"д
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         #:         #: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : : : : : : :	 :
 :  : 
╠-
Џ
F__inference_sequential_layer_call_and_return_conditional_losses_138234

lstm_input'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3)
%lstm_1_statefulpartitionedcall_args_1)
%lstm_1_statefulpartitionedcall_args_2)
%lstm_1_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdropout/StatefulPartitionedCallб!dropout_1/StatefulPartitionedCallб!dropout_2/StatefulPartitionedCallб!dropout_3/StatefulPartitionedCallбlstm/StatefulPartitionedCallбlstm_1/StatefulPartitionedCallЦ
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_input#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-137593*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_137421*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:         
#н
dropout/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:         
#*-
_gradient_op_typePartitionedCall-137644*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_137633*
Tout
2╔
lstm_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0%lstm_1_statefulpartitionedcall_args_1%lstm_1_statefulpartitionedcall_args_2%lstm_1_statefulpartitionedcall_args_3**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         #*-
_gradient_op_typePartitionedCall-137997*K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_137825*
Tout
2Э
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_138037*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         #*-
_gradient_op_typePartitionedCall-138048Ъ
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-138078*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_138072*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2щ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         #*-
_gradient_op_typePartitionedCall-138120*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_138109*
Tout
2Д
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-138150*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_138144*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         #ч
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2*-
_gradient_op_typePartitionedCall-138192*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_138181Д
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*'
_output_shapes
:         *
Tin
2*-
_gradient_op_typePartitionedCall-138222*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_138216*
Tout
2**
config_proto

GPU 

CPU2J 8б
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*Z
_input_shapesI
G:         
@::::::::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:* &
$
_user_specified_name
lstm_input: : : : : : : : :	 :
 : : 
і
Њ
while_cond_140512
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : ::::  : : : : : : : : :	 :
 
нP
А
@__inference_lstm_layer_call_and_return_conditional_losses_139875

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
dtype0*
_output_shapes
: *
value	B :#s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         #O
zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
dtype0*
_output_shapes
: *
value	B :#w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         #c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:
         @D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
         А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
T0*
Index0Б
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ї|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #]
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*'
_output_shapes
:         #*
T0_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         #*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         #*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #n
TensorArrayV2_1/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
_output_shapes
: *
value	B : *
dtype0Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
_output_shapes
: *
value	B : *
dtype0Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_139773*
_num_original_outputs*
bodyR
while_body_139774*L
_output_shapes:
8: : : : :         #:         #: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         #^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         #M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
valueB"    #   *
dtype0═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
         #h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         #*
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
#[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: «
IdentityIdentitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         
#"
identityIdentity:output:0*6
_input_shapes%
#:         
@:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs: : : 
▄
ћ
$__inference_signature_wrapper_138373

lstm_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tin
2*'
_output_shapes
:         *-
_gradient_op_typePartitionedCall-138358**
f%R#
!__inference__wrapped_model_135963*
Tout
2**
config_proto

GPU 

CPU2J 8ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*Z
_input_shapesI
G:         
@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :* &
$
_user_specified_name
lstm_input: : : : : : : : :	 :
 
П
Њ
while_cond_139421
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : ::::
 :  : : : : : : : : :	 
╩B
ь
@__inference_lstm_layer_call_and_return_conditional_losses_136460

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityѕбStatefulPartitionedCallбwhile;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0M
zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
_output_shapes
: *
value
B :У*
dtype0Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
_output_shapes
:*
T0*
NP
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         #O
zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         #c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  @D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
element_dtype0*
_output_shapes
: *

shape_type0_
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         @В
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-136079*N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_136038*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin

2*M
_output_shapes;
9:         #:         #:         #n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
valueB"    #   *
dtype0Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
_output_shapes
: *
valueB :
         *
dtype0T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: «
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5^StatefulPartitionedCall*
bodyR
while_body_136380*L
_output_shapes:
8: : : : :         #:         #: : : : : *
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_136379*
_num_original_outputsK
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         #^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         #M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  #h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
_output_shapes
:*
valueB:*
dtype0Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         #e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  #[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: є
IdentityIdentitytranspose_1:y:0^StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :                  #"
identityIdentity:output:0*?
_input_shapes.
,:                  @:::2
whilewhile22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
╗
c
*__inference_dropout_3_layer_call_fn_140766

inputs
identityѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputs*-
_gradient_op_typePartitionedCall-138192*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_138181*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         #ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*&
_input_shapes
:         #22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
нP
А
@__inference_lstm_layer_call_and_return_conditional_losses_139706

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         #O
zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         #c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:
         @*
T0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:         @*
Index0*
T0*
shrink_axis_maskБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ї|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Д
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0]
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         #*
T0n
TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
element_dtype0*
_output_shapes
: *

shape_type0F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_139604*
_num_original_outputs*
bodyR
while_body_139605*L
_output_shapes:
8: : : : :         #:         #: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:         #*
T0^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:         #*
T0M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
         #h
strided_slice_3/stackConst*
_output_shapes
:*
valueB:
         *
dtype0a
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         #e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*+
_output_shapes
:         
#*
T0[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: «
IdentityIdentitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         
#"
identityIdentity:output:0*6
_input_shapes%
#:         
@:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs: 
В+
з
while_body_137893
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         #Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Е
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0Б
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         #*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         #*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         #*
T0Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
dtype0*
_output_shapes
: *
value	B :N
add_2AddV2placeholderadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : : : : :	 :
 :  
▓
г
lstm_while_cond_138442
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_strided_slice_10
,lstm_tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
U
LessLessplaceholderless_lstm_strided_slice_1*
_output_shapes
: *
T0g
Less_1Lesslstm_while_loop_counterlstm_while_maximum_iterations*
_output_shapes
: *
T0F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : ::::
 :  : : : : : : : : :	 
┬
Х
lstm_1_while_cond_138623
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_1_strided_slice_12
.lstm_1_tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
W
LessLessplaceholderless_lstm_1_strided_slice_1*
T0*
_output_shapes
: k
Less_1Lesslstm_1_while_loop_counterlstm_1_while_maximum_iterations*
_output_shapes
: *
T0F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::: : : : : : : :	 :
 :  : 
▓,
ќ
lstm_while_body_138862
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_strided_slice_1_0V
Rtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_strided_slice_1T
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   Њ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	@ї*
dtype0ј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0Б
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: Z
add_3AddV2lstm_while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: А

Identity_1Identitylstm_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"д
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0".
lstm_strided_slice_1lstm_strided_slice_1_0*Q
_input_shapes@
>: : : : :         #:         #: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : :	 :
 :  : : : : : 
Џ
ь
#sequential_lstm_1_while_cond_135837(
$sequential_lstm_1_while_loop_counter.
*sequential_lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3*
&less_sequential_lstm_1_strided_slice_1=
9sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
b
LessLessplaceholder&less_sequential_lstm_1_strided_slice_1*
T0*
_output_shapes
: Ђ
Less_1Less$sequential_lstm_1_while_loop_counter*sequential_lstm_1_while_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::: : : : : :	 :
 :  : : : 
├
a
(__inference_dropout_layer_call_fn_139921

inputs
identityѕбStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:         
#*-
_gradient_op_typePartitionedCall-137644*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_137633*
Tout
2є
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*+
_output_shapes
:         
#*
T0"
identityIdentity:output:0**
_input_shapes
:         
#22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
о
Е
(__inference_dense_2_layer_call_fn_140789

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-138222*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_138216*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         #::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
љQ
Ц
B__inference_lstm_1_layer_call_and_return_conditional_losses_140260
inputs_0"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
_output_shapes
: *
value	B :#*
dtype0_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
_output_shapes
: *
T0P
zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:         #*
T0O
zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         #c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  #D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         #*
T0*
Index0Б
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	#ї*
dtype0|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #]
mulMulSigmoid_1:y:0zeros_1:output:0*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         #*
T0V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #n
TensorArrayV2_1/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
dtype0*
_output_shapes
: *
value	B : c
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
valueB :
         T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_num_original_outputs*
bodyR
while_body_140161*L
_output_shapes:
8: : : : :         #:         #: : : : : *K
output_shapes:
8: : : : :         #:         #: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_140160K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         #^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         #M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
valueB"    #   *
dtype0о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  #h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         #e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  #[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:         #*
T0"
identityIdentity:output:0*?
_input_shapes.
,:                  #:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:( $
"
_user_specified_name
inputs/0: : : 
╦
b
C__inference_dropout_layer_call_and_return_conditional_losses_139911

inputs
identityѕQ
dropout/rateConst*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*+
_output_shapes
:         
#ї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0д
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*+
_output_shapes
:         
#ў
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*+
_output_shapes
:         
#R
dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0Ї
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*+
_output_shapes
:         
#*
T0e
dropout/mulMulinputsdropout/truediv:z:0*
T0*+
_output_shapes
:         
#s
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*+
_output_shapes
:         
#m
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         
#]
IdentityIdentitydropout/mul_1:z:0*+
_output_shapes
:         
#*
T0"
identityIdentity:output:0**
_input_shapes
:         
#:& "
 
_user_specified_nameinputs
█P
Б
B__inference_lstm_1_layer_call_and_return_conditional_losses_140445

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0M
zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         #O
zeros_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :#c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
_output_shapes
:*
T0*
NR
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         #c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:
         #*
T0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         #Б
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#ї|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Д
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #]
mulMulSigmoid_1:y:0zeros_1:output:0*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*'
_output_shapes
:         #*
T0_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         #*
T0T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         #*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         #*
T0n
TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
_output_shapes
: *
value	B :
*
dtype0T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*K
output_shapes:
8: : : : :         #:         #: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_140343*
_num_original_outputs*
bodyR
while_body_140344*L
_output_shapes:
8: : : : :         #:         #: : : : : K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:         #*
T0^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         #M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   ═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
         #h
strided_slice_3/stackConst*
_output_shapes
:*
valueB:
         *
dtype0a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
_output_shapes
:*
valueB:*
dtype0Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         #e
transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
#[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*6
_input_shapes%
#:         
#:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs: : : 
н	
▄
C__inference_dense_2_layer_call_and_return_conditional_losses_140782

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:#i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0V
SoftmaxSoftmaxBiasAdd:output:0*'
_output_shapes
:         *
T0і
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         #::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
┐
D
(__inference_dropout_layer_call_fn_139926

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:         
#*
Tin
2*-
_gradient_op_typePartitionedCall-137652*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_137640d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         
#"
identityIdentity:output:0**
_input_shapes
:         
#:& "
 
_user_specified_nameinputs
і
Њ
while_cond_137319
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::: : :	 :
 :  : : : : : : 
Е
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_138181

inputs
identityѕQ
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *═╠L=C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:         #ї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0б
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:         #ћ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*'
_output_shapes
:         #*
T0R
dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: Ѕ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*'
_output_shapes
:         #*
T0a
dropout/mulMulinputsdropout/truediv:z:0*'
_output_shapes
:         #*
T0o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:         #i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*'
_output_shapes
:         #*
T0Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
Шј
Ѓ
F__inference_sequential_layer_call_and_return_conditional_losses_138794

inputs'
#lstm_matmul_readvariableop_resource)
%lstm_matmul_1_readvariableop_resource(
$lstm_biasadd_readvariableop_resource)
%lstm_1_matmul_readvariableop_resource+
'lstm_1_matmul_1_readvariableop_resource*
&lstm_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpбdense_2/BiasAdd/ReadVariableOpбdense_2/MatMul/ReadVariableOpбlstm/BiasAdd/ReadVariableOpбlstm/MatMul/ReadVariableOpбlstm/MatMul_1/ReadVariableOpб
lstm/whileбlstm_1/BiasAdd/ReadVariableOpбlstm_1/MatMul/ReadVariableOpбlstm_1/MatMul_1/ReadVariableOpбlstm_1/while@

lstm/ShapeShapeinputs*
_output_shapes
:*
T0b
lstm/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0d
lstm/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:d
lstm/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ж
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0R
lstm/zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: n
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: T
lstm/zeros/Less/yConst*
_output_shapes
: *
value
B :У*
dtype0h
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
_output_shapes
: *
T0U
lstm/zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: ѓ
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
_output_shapes
:*
T0U
lstm/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:         #T
lstm/zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: r
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
_output_shapes
: *
T0V
lstm/zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: n
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: W
lstm/zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: є
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:W
lstm/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: Ђ
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*'
_output_shapes
:         #*
T0h
lstm/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:w
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*+
_output_shapes
:
         @*
T0N
lstm/Shape_1Shapelstm/transpose:y:0*
_output_shapes
:*
T0d
lstm/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: f
lstm/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:f
lstm/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: k
 lstm/TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: ░
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: І
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:▄
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: d
lstm/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: f
lstm/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:f
lstm/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ѓ
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*'
_output_shapes
:         @*
Index0*
T0*
shrink_axis_maskГ
lstm/MatMul/ReadVariableOpReadVariableOp#lstm_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@їІ
lstm/MatMulMatMullstm/strided_slice_2:output:0"lstm/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ї▒
lstm/MatMul_1/ReadVariableOpReadVariableOp%lstm_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їЁ
lstm/MatMul_1MatMullstm/zeros:output:0$lstm/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їt
lstm/addAddV2lstm/MatMul:product:0lstm/MatMul_1:product:0*(
_output_shapes
:         ї*
T0Ф
lstm/BiasAdd/ReadVariableOpReadVariableOp$lstm_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ї}
lstm/BiasAddBiasAddlstm/add:z:0#lstm/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їL

lstm/ConstConst*
value	B :*
dtype0*
_output_shapes
: V
lstm/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ┼

lstm/splitSplitlstm/split/split_dim:output:0lstm/BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #^
lstm/SigmoidSigmoidlstm/split:output:0*
T0*'
_output_shapes
:         #`
lstm/Sigmoid_1Sigmoidlstm/split:output:1*'
_output_shapes
:         #*
T0l
lstm/mulMullstm/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:         #X
	lstm/ReluRelulstm/split:output:2*
T0*'
_output_shapes
:         #n

lstm/mul_1Mullstm/Sigmoid:y:0lstm/Relu:activations:0*'
_output_shapes
:         #*
T0c

lstm/add_1AddV2lstm/mul:z:0lstm/mul_1:z:0*'
_output_shapes
:         #*
T0`
lstm/Sigmoid_2Sigmoidlstm/split:output:3*
T0*'
_output_shapes
:         #U
lstm/Relu_1Relulstm/add_1:z:0*
T0*'
_output_shapes
:         #r

lstm/mul_2Mullstm/Sigmoid_2:y:0lstm/Relu_1:activations:0*
T0*'
_output_shapes
:         #s
"lstm/TensorArrayV2_1/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:┤
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: K
	lstm/timeConst*
value	B : *
dtype0*
_output_shapes
: _
lstm/while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: Y
lstm/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: и

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0#lstm_matmul_readvariableop_resource%lstm_matmul_1_readvariableop_resource$lstm_biasadd_readvariableop_resource^lstm/BiasAdd/ReadVariableOp^lstm/MatMul/ReadVariableOp^lstm/MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(*
parallel_iterations *"
condR
lstm_while_cond_138442*
_num_original_outputs*"
bodyR
lstm_while_body_138443*L
_output_shapes:
8: : : : :         #:         #: : : : : U
lstm/while/IdentityIdentitylstm/while:output:0*
_output_shapes
: *
T0W
lstm/while/Identity_1Identitylstm/while:output:1*
_output_shapes
: *
T0W
lstm/while/Identity_2Identitylstm/while:output:2*
T0*
_output_shapes
: W
lstm/while/Identity_3Identitylstm/while:output:3*
T0*
_output_shapes
: h
lstm/while/Identity_4Identitylstm/while:output:4*'
_output_shapes
:         #*
T0h
lstm/while/Identity_5Identitylstm/while:output:5*
T0*'
_output_shapes
:         #W
lstm/while/Identity_6Identitylstm/while:output:6*
_output_shapes
: *
T0W
lstm/while/Identity_7Identitylstm/while:output:7*
_output_shapes
: *
T0W
lstm/while/Identity_8Identitylstm/while:output:8*
_output_shapes
: *
T0W
lstm/while/Identity_9Identitylstm/while:output:9*
T0*
_output_shapes
: Y
lstm/while/Identity_10Identitylstm/while:output:10*
_output_shapes
: *
T0є
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:▄
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while/Identity_3:output:0>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
         #m
lstm/strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:f
lstm/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:f
lstm/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*'
_output_shapes
:         #*
T0*
Index0*
shrink_axis_maskj
lstm/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ц
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*+
_output_shapes
:         
#*
T0`
lstm/runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: Y
dropout/dropout/rateConst*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: Y
dropout/dropout/ShapeShapelstm/transpose_1:y:0*
T0*
_output_shapes
:g
"dropout/dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    g
"dropout/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: а
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
dtype0*+
_output_shapes
:         
#ц
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Й
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*+
_output_shapes
:         
#░
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*+
_output_shapes
:         
#Z
dropout/dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: ^
dropout/dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?ђ
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: Ц
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*+
_output_shapes
:         
#*
T0Ѓ
dropout/dropout/mulMullstm/transpose_1:y:0dropout/dropout/truediv:z:0*+
_output_shapes
:         
#*
T0Ѓ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*+
_output_shapes
:         
#Ё
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*+
_output_shapes
:         
#*
T0U
lstm_1/ShapeShapedropout/dropout/mul_1:z:0*
T0*
_output_shapes
:d
lstm_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:f
lstm_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:f
lstm_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskT
lstm_1/zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: t
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
_output_shapes
: *
T0V
lstm_1/zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: n
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
_output_shapes
: *
T0W
lstm_1/zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: ѕ
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:W
lstm_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: Ђ
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*'
_output_shapes
:         #*
T0V
lstm_1/zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: x
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: X
lstm_1/zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: t
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
_output_shapes
: *
T0Y
lstm_1/zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: ї
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0Y
lstm_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: Є
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*'
_output_shapes
:         #*
T0j
lstm_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:ј
lstm_1/transpose	Transposedropout/dropout/mul_1:z:0lstm_1/transpose/perm:output:0*
T0*+
_output_shapes
:
         #R
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:f
lstm_1/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: h
lstm_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:h
lstm_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:■
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskm
"lstm_1/TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Х
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ї
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
valueB"    #   *
dtype0Р
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: f
lstm_1/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:h
lstm_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:h
lstm_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ї
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         #*
T0*
Index0▒
lstm_1/MatMul/ReadVariableOpReadVariableOp%lstm_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їЉ
lstm_1/MatMulMatMullstm_1/strided_slice_2:output:0$lstm_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їх
lstm_1/MatMul_1/ReadVariableOpReadVariableOp'lstm_1_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їІ
lstm_1/MatMul_1MatMullstm_1/zeros:output:0&lstm_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їz

lstm_1/addAddV2lstm_1/MatMul:product:0lstm_1/MatMul_1:product:0*(
_output_shapes
:         ї*
T0»
lstm_1/BiasAdd/ReadVariableOpReadVariableOp&lstm_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їЃ
lstm_1/BiasAddBiasAddlstm_1/add:z:0%lstm_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їN
lstm_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: X
lstm_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ╦
lstm_1/splitSplitlstm_1/split/split_dim:output:0lstm_1/BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0b
lstm_1/SigmoidSigmoidlstm_1/split:output:0*'
_output_shapes
:         #*
T0d
lstm_1/Sigmoid_1Sigmoidlstm_1/split:output:1*
T0*'
_output_shapes
:         #r

lstm_1/mulMullstm_1/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:         #\
lstm_1/ReluRelulstm_1/split:output:2*
T0*'
_output_shapes
:         #t
lstm_1/mul_1Mullstm_1/Sigmoid:y:0lstm_1/Relu:activations:0*'
_output_shapes
:         #*
T0i
lstm_1/add_1AddV2lstm_1/mul:z:0lstm_1/mul_1:z:0*'
_output_shapes
:         #*
T0d
lstm_1/Sigmoid_2Sigmoidlstm_1/split:output:3*
T0*'
_output_shapes
:         #Y
lstm_1/Relu_1Relulstm_1/add_1:z:0*
T0*'
_output_shapes
:         #x
lstm_1/mul_2Mullstm_1/Sigmoid_2:y:0lstm_1/Relu_1:activations:0*
T0*'
_output_shapes
:         #u
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
valueB"    #   *
dtype0║
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: M
lstm_1/timeConst*
value	B : *
dtype0*
_output_shapes
: a
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
value	B :
*
dtype0[
lstm_1/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ┘
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0%lstm_1_matmul_readvariableop_resource'lstm_1_matmul_1_readvariableop_resource&lstm_1_biasadd_readvariableop_resource^lstm_1/BiasAdd/ReadVariableOp^lstm_1/MatMul/ReadVariableOp^lstm_1/MatMul_1/ReadVariableOp*
parallel_iterations *$
condR
lstm_1_while_cond_138623*
_num_original_outputs*$
bodyR
lstm_1_while_body_138624*L
_output_shapes:
8: : : : :         #:         #: : : : : *
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(Y
lstm_1/while/IdentityIdentitylstm_1/while:output:0*
T0*
_output_shapes
: [
lstm_1/while/Identity_1Identitylstm_1/while:output:1*
T0*
_output_shapes
: [
lstm_1/while/Identity_2Identitylstm_1/while:output:2*
_output_shapes
: *
T0[
lstm_1/while/Identity_3Identitylstm_1/while:output:3*
_output_shapes
: *
T0l
lstm_1/while/Identity_4Identitylstm_1/while:output:4*'
_output_shapes
:         #*
T0l
lstm_1/while/Identity_5Identitylstm_1/while:output:5*'
_output_shapes
:         #*
T0[
lstm_1/while/Identity_6Identitylstm_1/while:output:6*
_output_shapes
: *
T0[
lstm_1/while/Identity_7Identitylstm_1/while:output:7*
_output_shapes
: *
T0[
lstm_1/while/Identity_8Identitylstm_1/while:output:8*
T0*
_output_shapes
: [
lstm_1/while/Identity_9Identitylstm_1/while:output:9*
T0*
_output_shapes
: ]
lstm_1/while/Identity_10Identitylstm_1/while:output:10*
_output_shapes
: *
T0ѕ
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   Р
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack lstm_1/while/Identity_3:output:0@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
         #o
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
valueB:
         *
dtype0h
lstm_1/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:h
lstm_1/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ф
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         #*
Index0*
T0l
lstm_1/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ф
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*+
_output_shapes
:         
#*
T0b
lstm_1/runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: [
dropout_1/dropout/rateConst*
valueB
 *═╠L=*
dtype0*
_output_shapes
: f
dropout_1/dropout/ShapeShapelstm_1/strided_slice_3:output:0*
T0*
_output_shapes
:i
$dropout_1/dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0i
$dropout_1/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: а
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
dtype0*'
_output_shapes
:         #*
T0ф
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0└
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:         #▓
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*'
_output_shapes
:         #*
T0\
dropout_1/dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ђ?*
dtype0ђ
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_1/dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: є
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
_output_shapes
: *
T0Д
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*
T0*'
_output_shapes
:         #ј
dropout_1/dropout/mulMullstm_1/strided_slice_3:output:0dropout_1/dropout/truediv:z:0*'
_output_shapes
:         #*
T0Ѓ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:         #Є
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*'
_output_shapes
:         #*
T0«
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:##і
dense/MatMulMatMuldropout_1/dropout/mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         #*
T0г
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:#ѕ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         #\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         #[
dropout_2/dropout/rateConst*
valueB
 *═╠L=*
dtype0*
_output_shapes
: _
dropout_2/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:i
$dropout_2/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_2/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: а
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*'
_output_shapes
:         #*
T0*
dtype0ф
$dropout_2/dropout/random_uniform/subSub-dropout_2/dropout/random_uniform/max:output:0-dropout_2/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0└
$dropout_2/dropout/random_uniform/mulMul7dropout_2/dropout/random_uniform/RandomUniform:output:0(dropout_2/dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:         #▓
 dropout_2/dropout/random_uniformAdd(dropout_2/dropout/random_uniform/mul:z:0-dropout_2/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:         #\
dropout_2/dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: ђ
dropout_2/dropout/subSub dropout_2/dropout/sub/x:output:0dropout_2/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_2/dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: є
dropout_2/dropout/truedivRealDiv$dropout_2/dropout/truediv/x:output:0dropout_2/dropout/sub:z:0*
T0*
_output_shapes
: Д
dropout_2/dropout/GreaterEqualGreaterEqual$dropout_2/dropout/random_uniform:z:0dropout_2/dropout/rate:output:0*
T0*'
_output_shapes
:         #Є
dropout_2/dropout/mulMuldense/Relu:activations:0dropout_2/dropout/truediv:z:0*
T0*'
_output_shapes
:         #Ѓ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:         #Є
dropout_2/dropout/mul_1Muldropout_2/dropout/mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:         #▓
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:##ј
dense_1/MatMulMatMuldropout_2/dropout/mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         #░
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:#ј
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         #`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         #[
dropout_3/dropout/rateConst*
valueB
 *═╠L=*
dtype0*
_output_shapes
: a
dropout_3/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:i
$dropout_3/dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    i
$dropout_3/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: а
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:         #ф
$dropout_3/dropout/random_uniform/subSub-dropout_3/dropout/random_uniform/max:output:0-dropout_3/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: └
$dropout_3/dropout/random_uniform/mulMul7dropout_3/dropout/random_uniform/RandomUniform:output:0(dropout_3/dropout/random_uniform/sub:z:0*'
_output_shapes
:         #*
T0▓
 dropout_3/dropout/random_uniformAdd(dropout_3/dropout/random_uniform/mul:z:0-dropout_3/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:         #\
dropout_3/dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: ђ
dropout_3/dropout/subSub dropout_3/dropout/sub/x:output:0dropout_3/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_3/dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: є
dropout_3/dropout/truedivRealDiv$dropout_3/dropout/truediv/x:output:0dropout_3/dropout/sub:z:0*
T0*
_output_shapes
: Д
dropout_3/dropout/GreaterEqualGreaterEqual$dropout_3/dropout/random_uniform:z:0dropout_3/dropout/rate:output:0*
T0*'
_output_shapes
:         #Ѕ
dropout_3/dropout/mulMuldense_1/Relu:activations:0dropout_3/dropout/truediv:z:0*
T0*'
_output_shapes
:         #Ѓ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:         #Є
dropout_3/dropout/mul_1Muldropout_3/dropout/mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:         #▓
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:#ј
dense_2/MatMulMatMuldropout_3/dropout/mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0░
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:ј
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*'
_output_shapes
:         *
T0Ш
IdentityIdentitydense_2/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^lstm/BiasAdd/ReadVariableOp^lstm/MatMul/ReadVariableOp^lstm/MatMul_1/ReadVariableOp^lstm/while^lstm_1/BiasAdd/ReadVariableOp^lstm_1/MatMul/ReadVariableOp^lstm_1/MatMul_1/ReadVariableOp^lstm_1/while*'
_output_shapes
:         *
T0"
identityIdentity:output:0*Z
_input_shapesI
G:         
@::::::::::::2>
lstm_1/BiasAdd/ReadVariableOplstm_1/BiasAdd/ReadVariableOp28
lstm/MatMul/ReadVariableOplstm/MatMul/ReadVariableOp2<
lstm/MatMul_1/ReadVariableOplstm/MatMul_1/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2:
lstm/BiasAdd/ReadVariableOplstm/BiasAdd/ReadVariableOp2

lstm/while
lstm/while2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2<
lstm_1/MatMul/ReadVariableOplstm_1/MatMul/ReadVariableOp2@
lstm_1/MatMul_1/ReadVariableOplstm_1/MatMul_1/ReadVariableOp2
lstm_1/whilelstm_1/while2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
╗
c
*__inference_dropout_2_layer_call_fn_140713

inputs
identityѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2*-
_gradient_op_typePartitionedCall-138120*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_138109*
Tout
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*&
_input_shapes
:         #22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
П
Њ
while_cond_139254
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::: : : : : : :	 :
 :  : : 
ѕ
Џ
+__inference_sequential_layer_call_fn_138350

lstm_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityѕбStatefulPartitionedCall┐
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         *-
_gradient_op_typePartitionedCall-138335*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_138334ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*Z
_input_shapesI
G:         
@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :* &
$
_user_specified_name
lstm_input: : : : : : : : :	 :
 : 
В+
з
while_body_140344
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
valueB"    #   *
dtype0ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         #Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Е
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0Б
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         #*
T0Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
_output_shapes
: *
value	B :*
dtype0N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 :
 
њQ
Б
@__inference_lstm_layer_call_and_return_conditional_losses_139521
inputs_0"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
dtype0*
_output_shapes
: *
value	B :#_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
dtype0*
_output_shapes
: *
value
B :УY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:         #*
T0O
zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         #c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  @D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         @Б
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	@ї*
dtype0|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Д
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0]
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #n
TensorArrayV2_1/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
_output_shapes
: *
valueB :
         *
dtype0T
while/loop_counterConst*
_output_shapes
: *
value	B : *
dtype0Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_139422*L
_output_shapes:
8: : : : :         #:         #: : : : : *
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_139421*
_num_original_outputsK
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         #^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         #M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  #h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*'
_output_shapes
:         #*
Index0*
T0*
shrink_axis_maske
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :                  #*
T0[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: и
IdentityIdentitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :                  #"
identityIdentity:output:0*?
_input_shapes.
,:                  @:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:( $
"
_user_specified_name
inputs/0: : : 
і
Њ
while_cond_137488
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : ::::
 :  : : : : : : : : :	 
Е
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_138037

inputs
identityѕQ
dropout/rateConst*
valueB
 *═╠L=*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ђ?*
dtype0ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:         #ї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: б
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:         #ћ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:         #R
dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0Ѕ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:         #a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:         #o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:         #i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         #Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
╩B
ь
@__inference_lstm_layer_call_and_return_conditional_losses_136600

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityѕбStatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskM
zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:         #*
T0O
zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         #c
transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  @D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:         @*
Index0*
T0*
shrink_axis_maskВ
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin

2*M
_output_shapes;
9:         #:         #:         #*-
_gradient_op_typePartitionedCall-136096*N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_136073n
TensorArrayV2_1/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
dtype0*
_output_shapes
: *
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
valueB :
         *
dtype0T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: «
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5^StatefulPartitionedCall*
_num_original_outputs*
bodyR
while_body_136520*L
_output_shapes:
8: : : : :         #:         #: : : : : *K
output_shapes:
8: : : : :         #:         #: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_136519K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         #^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         #M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  #h
strided_slice_3/stackConst*
_output_shapes
:*
valueB:
         *
dtype0a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         #*
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :                  #*
T0[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: є
IdentityIdentitytranspose_1:y:0^StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :                  #"
identityIdentity:output:0*?
_input_shapes.
,:                  @:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile: :& "
 
_user_specified_nameinputs: : 
ы
І
*__inference_lstm_cell_layer_call_fn_140883

inputs
states_0
states_1"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity

identity_1

identity_2ѕбStatefulPartitionedCall╠
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*-
_gradient_op_typePartitionedCall-136096*N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_136073*
Tout
2**
config_proto

GPU 

CPU2J 8*M
_output_shapes;
9:         #:         #:         #*
Tin

2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         #ё

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         #ё

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*'
_output_shapes
:         #*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*X
_input_shapesG
E:         @:         #:         #:::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1
ш
Ї
,__inference_lstm_cell_1_layer_call_fn_140963

inputs
states_0
states_1"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity

identity_1

identity_2ѕбStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5**
config_proto

GPU 

CPU2J 8*M
_output_shapes;
9:         #:         #:         #*
Tin

2*-
_gradient_op_typePartitionedCall-136721*P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_136680*
Tout
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         #ё

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         #ё

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         #"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*X
_input_shapesG
E:         #:         #:         #:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : 
└-
Ќ
F__inference_sequential_layer_call_and_return_conditional_losses_138289

inputs'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3)
%lstm_1_statefulpartitionedcall_args_1)
%lstm_1_statefulpartitionedcall_args_2)
%lstm_1_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdropout/StatefulPartitionedCallб!dropout_1/StatefulPartitionedCallб!dropout_2/StatefulPartitionedCallб!dropout_3/StatefulPartitionedCallбlstm/StatefulPartitionedCallбlstm_1/StatefulPartitionedCallА
lstm/StatefulPartitionedCallStatefulPartitionedCallinputs#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:         
#*
Tin
2*-
_gradient_op_typePartitionedCall-137593*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_137421*
Tout
2н
dropout/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:         
#*
Tin
2*-
_gradient_op_typePartitionedCall-137644*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_137633*
Tout
2╔
lstm_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0%lstm_1_statefulpartitionedcall_args_1%lstm_1_statefulpartitionedcall_args_2%lstm_1_statefulpartitionedcall_args_3**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2*-
_gradient_op_typePartitionedCall-137997*K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_137825*
Tout
2Э
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         #*-
_gradient_op_typePartitionedCall-138048*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_138037*
Tout
2Ъ
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_138072*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2*-
_gradient_op_typePartitionedCall-138078щ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*-
_gradient_op_typePartitionedCall-138120*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_138109*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2Д
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-138150*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_138144*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2ч
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*'
_output_shapes
:         #*
Tin
2*-
_gradient_op_typePartitionedCall-138192*N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_138181*
Tout
2**
config_proto

GPU 

CPU2J 8Д
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         *
Tin
2*-
_gradient_op_typePartitionedCall-138222*L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_138216*
Tout
2б
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*Z
_input_shapesI
G:         
@::::::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
█P
Б
B__inference_lstm_1_layer_call_and_return_conditional_losses_137994

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
value
B :У*
dtype0Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         #O
zeros_1/mul/yConst*
_output_shapes
: *
value	B :#*
dtype0c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         #c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:
         #*
T0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
_output_shapes
: *
valueB :
         *
dtype0А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
valueB"    #   *
dtype0═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         #Б
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#ї|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	#ї*
dtype0v
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #]
mulMulSigmoid_1:y:0zeros_1:output:0*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         #*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #n
TensorArrayV2_1/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
_output_shapes
: *
value	B : *
dtype0Z
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
value	B :
T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_num_original_outputs*
bodyR
while_body_137893*L
_output_shapes:
8: : : : :         #:         #: : : : : *
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_137892K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         #^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         #M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
         #h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         #e
transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
#[
runtimeConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *    │
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:         #*
T0"
identityIdentity:output:0*6
_input_shapes%
#:         
#:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs: : : 
В+
з
while_body_139994
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         #Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         #*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         #*
T0Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 :
 
Ф
╩
%__inference_lstm_layer_call_fn_139883

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-137593*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_137421*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:         
#*
Tin
2є
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         
#"
identityIdentity:output:0*6
_input_shapes%
#:         
@:::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs
╠
╠
%__inference_lstm_layer_call_fn_139529
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3**
config_proto

GPU 

CPU2J 8*
Tin
2*4
_output_shapes"
 :                  #*-
_gradient_op_typePartitionedCall-136461*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_136460*
Tout
2Ј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  #"
identityIdentity:output:0*?
_input_shapes.
,:                  @:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0: : : 
╩B
№
B__inference_lstm_1_layer_call_and_return_conditional_losses_137242

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityѕбStatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0M
zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         #O
zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         #c
transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*4
_output_shapes"
 :                  #*
T0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   ═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         #Ь
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*M
_output_shapes;
9:         #:         #:         #*
Tin

2*-
_gradient_op_typePartitionedCall-136738*P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_136715*
Tout
2**
config_proto

GPU 

CPU2J 8n
TensorArrayV2_1/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
         *
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: «
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5^StatefulPartitionedCall*
parallel_iterations *
condR
while_cond_137161*
_num_original_outputs*
bodyR
while_body_137162*L
_output_shapes:
8: : : : :         #:         #: : : : : *
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:         #*
T0^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:         #*
T0M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  #h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         #*
Index0*
T0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :                  #*
T0[
runtimeConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *    ѓ
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*?
_input_shapes.
,:                  #:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile: : : :& "
 
_user_specified_nameinputs
Х
╬
'__inference_lstm_1_layer_call_fn_140276
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*'
_output_shapes
:         #*-
_gradient_op_typePartitionedCall-137243*K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_137242*
Tout
2**
config_proto

GPU 

CPU2J 8ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         #*
T0"
identityIdentity:output:0*?
_input_shapes.
,:                  #:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0: : : 
џ¤
Ѓ
F__inference_sequential_layer_call_and_return_conditional_losses_139153

inputs'
#lstm_matmul_readvariableop_resource)
%lstm_matmul_1_readvariableop_resource(
$lstm_biasadd_readvariableop_resource)
%lstm_1_matmul_readvariableop_resource+
'lstm_1_matmul_1_readvariableop_resource*
&lstm_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpбdense_2/BiasAdd/ReadVariableOpбdense_2/MatMul/ReadVariableOpбlstm/BiasAdd/ReadVariableOpбlstm/MatMul/ReadVariableOpбlstm/MatMul_1/ReadVariableOpб
lstm/whileбlstm_1/BiasAdd/ReadVariableOpбlstm_1/MatMul/ReadVariableOpбlstm_1/MatMul_1/ReadVariableOpбlstm_1/while@

lstm/ShapeShapeinputs*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:d
lstm/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:d
lstm/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ж
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: R
lstm/zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: n
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: T
lstm/zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: h
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
_output_shapes
: *
T0U
lstm/zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: ѓ
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:U
lstm/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:         #T
lstm/zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: r
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: V
lstm/zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: n
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
_output_shapes
: *
T0W
lstm/zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: є
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:W
lstm/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: Ђ
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:         #h
lstm/transpose/permConst*
_output_shapes
:*!
valueB"          *
dtype0w
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*+
_output_shapes
:
         @*
T0N
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: f
lstm/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:f
lstm/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0k
 lstm/TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: ░
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: І
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
valueB"    @   *
dtype0▄
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: d
lstm/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
valueB:*
dtype0f
lstm/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ѓ
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
Index0*
T0Г
lstm/MatMul/ReadVariableOpReadVariableOp#lstm_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@їІ
lstm/MatMulMatMullstm/strided_slice_2:output:0"lstm/MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0▒
lstm/MatMul_1/ReadVariableOpReadVariableOp%lstm_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їЁ
lstm/MatMul_1MatMullstm/zeros:output:0$lstm/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їt
lstm/addAddV2lstm/MatMul:product:0lstm/MatMul_1:product:0*
T0*(
_output_shapes
:         їФ
lstm/BiasAdd/ReadVariableOpReadVariableOp$lstm_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ї}
lstm/BiasAddBiasAddlstm/add:z:0#lstm/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їL

lstm/ConstConst*
value	B :*
dtype0*
_output_shapes
: V
lstm/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ┼

lstm/splitSplitlstm/split/split_dim:output:0lstm/BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #^
lstm/SigmoidSigmoidlstm/split:output:0*
T0*'
_output_shapes
:         #`
lstm/Sigmoid_1Sigmoidlstm/split:output:1*'
_output_shapes
:         #*
T0l
lstm/mulMullstm/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:         #X
	lstm/ReluRelulstm/split:output:2*
T0*'
_output_shapes
:         #n

lstm/mul_1Mullstm/Sigmoid:y:0lstm/Relu:activations:0*
T0*'
_output_shapes
:         #c

lstm/add_1AddV2lstm/mul:z:0lstm/mul_1:z:0*
T0*'
_output_shapes
:         #`
lstm/Sigmoid_2Sigmoidlstm/split:output:3*'
_output_shapes
:         #*
T0U
lstm/Relu_1Relulstm/add_1:z:0*
T0*'
_output_shapes
:         #r

lstm/mul_2Mullstm/Sigmoid_2:y:0lstm/Relu_1:activations:0*
T0*'
_output_shapes
:         #s
"lstm/TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   ┤
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: K
	lstm/timeConst*
value	B : *
dtype0*
_output_shapes
: _
lstm/while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: Y
lstm/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: и

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0#lstm_matmul_readvariableop_resource%lstm_matmul_1_readvariableop_resource$lstm_biasadd_readvariableop_resource^lstm/BiasAdd/ReadVariableOp^lstm/MatMul/ReadVariableOp^lstm/MatMul_1/ReadVariableOp*
_num_original_outputs*"
bodyR
lstm_while_body_138862*L
_output_shapes:
8: : : : :         #:         #: : : : : *
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(*
parallel_iterations *"
condR
lstm_while_cond_138861U
lstm/while/IdentityIdentitylstm/while:output:0*
_output_shapes
: *
T0W
lstm/while/Identity_1Identitylstm/while:output:1*
_output_shapes
: *
T0W
lstm/while/Identity_2Identitylstm/while:output:2*
T0*
_output_shapes
: W
lstm/while/Identity_3Identitylstm/while:output:3*
T0*
_output_shapes
: h
lstm/while/Identity_4Identitylstm/while:output:4*
T0*'
_output_shapes
:         #h
lstm/while/Identity_5Identitylstm/while:output:5*'
_output_shapes
:         #*
T0W
lstm/while/Identity_6Identitylstm/while:output:6*
_output_shapes
: *
T0W
lstm/while/Identity_7Identitylstm/while:output:7*
T0*
_output_shapes
: W
lstm/while/Identity_8Identitylstm/while:output:8*
_output_shapes
: *
T0W
lstm/while/Identity_9Identitylstm/while:output:9*
T0*
_output_shapes
: Y
lstm/while/Identity_10Identitylstm/while:output:10*
T0*
_output_shapes
: є
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:▄
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while/Identity_3:output:0>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
         #m
lstm/strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:f
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
valueB: *
dtype0f
lstm/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:а
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         #j
lstm/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ц
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:         
#`
lstm/runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: h
dropout/IdentityIdentitylstm/transpose_1:y:0*
T0*+
_output_shapes
:         
#U
lstm_1/ShapeShapedropout/Identity:output:0*
T0*
_output_shapes
:d
lstm_1/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0f
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0f
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0З
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0T
lstm_1/zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: t
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: V
lstm_1/zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: n
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
_output_shapes
: *
T0W
lstm_1/zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: ѕ
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:W
lstm_1/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0Ђ
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:         #V
lstm_1/zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: x
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
_output_shapes
: *
T0X
lstm_1/zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: t
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: Y
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
value	B :#*
dtype0ї
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:Y
lstm_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: Є
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*'
_output_shapes
:         #*
T0j
lstm_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:ј
lstm_1/transpose	Transposedropout/Identity:output:0lstm_1/transpose/perm:output:0*+
_output_shapes
:
         #*
T0R
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:f
lstm_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:h
lstm_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:h
lstm_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:■
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: m
"lstm_1/TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Х
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ї
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Р
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: f
lstm_1/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:h
lstm_1/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:h
lstm_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ї
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         #*
T0*
Index0▒
lstm_1/MatMul/ReadVariableOpReadVariableOp%lstm_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	#ї*
dtype0Љ
lstm_1/MatMulMatMullstm_1/strided_slice_2:output:0$lstm_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їх
lstm_1/MatMul_1/ReadVariableOpReadVariableOp'lstm_1_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їІ
lstm_1/MatMul_1MatMullstm_1/zeros:output:0&lstm_1/MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0z

lstm_1/addAddV2lstm_1/MatMul:product:0lstm_1/MatMul_1:product:0*
T0*(
_output_shapes
:         ї»
lstm_1/BiasAdd/ReadVariableOpReadVariableOp&lstm_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їЃ
lstm_1/BiasAddBiasAddlstm_1/add:z:0%lstm_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їN
lstm_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: X
lstm_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ╦
lstm_1/splitSplitlstm_1/split/split_dim:output:0lstm_1/BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #b
lstm_1/SigmoidSigmoidlstm_1/split:output:0*'
_output_shapes
:         #*
T0d
lstm_1/Sigmoid_1Sigmoidlstm_1/split:output:1*
T0*'
_output_shapes
:         #r

lstm_1/mulMullstm_1/Sigmoid_1:y:0lstm_1/zeros_1:output:0*'
_output_shapes
:         #*
T0\
lstm_1/ReluRelulstm_1/split:output:2*
T0*'
_output_shapes
:         #t
lstm_1/mul_1Mullstm_1/Sigmoid:y:0lstm_1/Relu:activations:0*
T0*'
_output_shapes
:         #i
lstm_1/add_1AddV2lstm_1/mul:z:0lstm_1/mul_1:z:0*
T0*'
_output_shapes
:         #d
lstm_1/Sigmoid_2Sigmoidlstm_1/split:output:3*
T0*'
_output_shapes
:         #Y
lstm_1/Relu_1Relulstm_1/add_1:z:0*
T0*'
_output_shapes
:         #x
lstm_1/mul_2Mullstm_1/Sigmoid_2:y:0lstm_1/Relu_1:activations:0*
T0*'
_output_shapes
:         #u
$lstm_1/TensorArrayV2_1/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:║
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: M
lstm_1/timeConst*
dtype0*
_output_shapes
: *
value	B : a
lstm_1/while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: [
lstm_1/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ┘
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0%lstm_1_matmul_readvariableop_resource'lstm_1_matmul_1_readvariableop_resource&lstm_1_biasadd_readvariableop_resource^lstm_1/BiasAdd/ReadVariableOp^lstm_1/MatMul/ReadVariableOp^lstm_1/MatMul_1/ReadVariableOp*$
condR
lstm_1_while_cond_139027*
_num_original_outputs*$
bodyR
lstm_1_while_body_139028*L
_output_shapes:
8: : : : :         #:         #: : : : : *
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(*
parallel_iterations Y
lstm_1/while/IdentityIdentitylstm_1/while:output:0*
_output_shapes
: *
T0[
lstm_1/while/Identity_1Identitylstm_1/while:output:1*
_output_shapes
: *
T0[
lstm_1/while/Identity_2Identitylstm_1/while:output:2*
_output_shapes
: *
T0[
lstm_1/while/Identity_3Identitylstm_1/while:output:3*
_output_shapes
: *
T0l
lstm_1/while/Identity_4Identitylstm_1/while:output:4*
T0*'
_output_shapes
:         #l
lstm_1/while/Identity_5Identitylstm_1/while:output:5*'
_output_shapes
:         #*
T0[
lstm_1/while/Identity_6Identitylstm_1/while:output:6*
_output_shapes
: *
T0[
lstm_1/while/Identity_7Identitylstm_1/while:output:7*
_output_shapes
: *
T0[
lstm_1/while/Identity_8Identitylstm_1/while:output:8*
_output_shapes
: *
T0[
lstm_1/while/Identity_9Identitylstm_1/while:output:9*
T0*
_output_shapes
: ]
lstm_1/while/Identity_10Identitylstm_1/while:output:10*
T0*
_output_shapes
: ѕ
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Р
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack lstm_1/while/Identity_3:output:0@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
         #o
lstm_1/strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:h
lstm_1/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: h
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
valueB:*
dtype0ф
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         #l
lstm_1/transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          Ф
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:         
#b
lstm_1/runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: q
dropout_1/IdentityIdentitylstm_1/strided_slice_3:output:0*
T0*'
_output_shapes
:         #«
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:##*
dtype0і
dense/MatMulMatMuldropout_1/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         #г
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:#ѕ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         #\

dense/ReluReludense/BiasAdd:output:0*'
_output_shapes
:         #*
T0j
dropout_2/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:         #▓
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:##ј
dense_1/MatMulMatMuldropout_2/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         #*
T0░
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:#ј
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         #*
T0`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         #l
dropout_3/IdentityIdentitydense_1/Relu:activations:0*'
_output_shapes
:         #*
T0▓
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:#ј
dense_2/MatMulMatMuldropout_3/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ░
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:ј
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0f
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*'
_output_shapes
:         *
T0Ш
IdentityIdentitydense_2/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^lstm/BiasAdd/ReadVariableOp^lstm/MatMul/ReadVariableOp^lstm/MatMul_1/ReadVariableOp^lstm/while^lstm_1/BiasAdd/ReadVariableOp^lstm_1/MatMul/ReadVariableOp^lstm_1/MatMul_1/ReadVariableOp^lstm_1/while*'
_output_shapes
:         *
T0"
identityIdentity:output:0*Z
_input_shapesI
G:         
@::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2<
lstm_1/MatMul/ReadVariableOplstm_1/MatMul/ReadVariableOp2@
lstm_1/MatMul_1/ReadVariableOplstm_1/MatMul_1/ReadVariableOp2
lstm_1/whilelstm_1/while2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
lstm/MatMul/ReadVariableOplstm/MatMul/ReadVariableOp2>
lstm_1/BiasAdd/ReadVariableOplstm_1/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2:
lstm/BiasAdd/ReadVariableOplstm/BiasAdd/ReadVariableOp2<
lstm/MatMul_1/ReadVariableOplstm/MatMul_1/ReadVariableOp2

lstm/while
lstm/while:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
В+
з
while_body_139255
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0Z
mulMulSigmoid_1:y:0placeholder_3*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:  : : : : : : : : :	 :
 
В+
з
while_body_140161
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         #Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:ї*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*'
_output_shapes
:         #*
T0_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         #*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         #*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         #*
T0Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
_output_shapes
: *
value	B :*
dtype0N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         #*
T0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0*Q
_input_shapes@
>: : : : :         #:         #: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:  : : : : : : : : :	 :
 
и
F
*__inference_dropout_2_layer_call_fn_140718

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*'
_output_shapes
:         #*-
_gradient_op_typePartitionedCall-138128*N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_138116*
Tout
2**
config_proto

GPU 

CPU2J 8`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
Џ
╝
while_body_136380
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0$
 statefulpartitionedcall_args_5_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5ѕбStatefulPartitionedCallѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @ђ
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0 statefulpartitionedcall_args_5_0*-
_gradient_op_typePartitionedCall-136079*N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_136038*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin

2*M
_output_shapes;
9:         #:         #:         #ц
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
element_dtype0*
_output_shapes
: G
add/yConst*
value	B :*
dtype0*
_output_shapes
: J
addAddV2placeholderadd/y:output:0*
_output_shapes
: *
T0I
add_1/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: Z
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: k

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: Z

Identity_2Identityadd:z:0^StatefulPartitionedCall*
_output_shapes
: *
T0Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: ё

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*'
_output_shapes
:         #*
T0ё

Identity_5Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         #"B
statefulpartitionedcall_args_5 statefulpartitionedcall_args_5_0"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0*Q
_input_shapes@
>: : : : :         #:         #: : :::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :	 :
 :  : : : : 
і
Њ
while_cond_140343
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
_output_shapes
: *
T0F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : ::::
 :  : : : : : : : : :	 
Ч
Ќ
+__inference_sequential_layer_call_fn_139170

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityѕбStatefulPartitionedCall╗
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*-
_gradient_op_typePartitionedCall-138290*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_138289*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*Z
_input_shapesI
G:         
@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
Џ
╝
while_body_136520
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0$
 statefulpartitionedcall_args_5_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5ѕбStatefulPartitionedCallѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @ђ
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0 statefulpartitionedcall_args_5_0*-
_gradient_op_typePartitionedCall-136096*N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_136073*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin

2*M
_output_shapes;
9:         #:         #:         #ц
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
element_dtype0*
_output_shapes
: G
add/yConst*
value	B :*
dtype0*
_output_shapes
: J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_1AddV2while_loop_counteradd_1/y:output:0*
_output_shapes
: *
T0Z
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
_output_shapes
: *
T0k

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: Z

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: ё

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         #ё

Identity_5Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         #"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0"B
statefulpartitionedcall_args_5 statefulpartitionedcall_args_5_0"$
strided_slice_1strided_slice_1_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0*Q
_input_shapes@
>: : : : :         #:         #: : :::22
StatefulPartitionedCallStatefulPartitionedCall: : : :	 :
 :  : : : : : 
Ю
╝
while_body_137022
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0$
 statefulpartitionedcall_args_5_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5ѕбStatefulPartitionedCallѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         #ѓ
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0 statefulpartitionedcall_args_5_0*-
_gradient_op_typePartitionedCall-136721*P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_136680*
Tout
2**
config_proto

GPU 

CPU2J 8*M
_output_shapes;
9:         #:         #:         #*
Tin

2ц
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
element_dtype0*
_output_shapes
: G
add/yConst*
dtype0*
_output_shapes
: *
value	B :J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_1AddV2while_loop_counteradd_1/y:output:0*
_output_shapes
: *
T0Z
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
_output_shapes
: *
T0k

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: Z

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: ё

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         #ё

Identity_5Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         #"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0"B
statefulpartitionedcall_args_5 statefulpartitionedcall_args_5_0"$
strided_slice_1strided_slice_1_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::22
StatefulPartitionedCallStatefulPartitionedCall:  : : : : : : : : :	 :
 
═	
┌
A__inference_dense_layer_call_and_return_conditional_losses_138072

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:##i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         #а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:#v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         #*
T0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         #І
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         #*
T0"
identityIdentity:output:0*.
_input_shapes
:         #::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
П
Њ
while_cond_136519
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : ::::  : : : : : : : : :	 :
 
П
Њ
while_cond_140160
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::: : : : : : :	 :
 :  : : 
и
F
*__inference_dropout_1_layer_call_fn_140665

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2*-
_gradient_op_typePartitionedCall-138056*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_138044`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*&
_input_shapes
:         #:& "
 
_user_specified_nameinputs
Д
╠
'__inference_lstm_1_layer_call_fn_140622

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*'
_output_shapes
:         #*
Tin
2*-
_gradient_op_typePartitionedCall-137997*K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_137825*
Tout
2**
config_proto

GPU 

CPU2J 8ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         #*
T0"
identityIdentity:output:0*6
_input_shapes%
#:         
#:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
▓
г
lstm_while_cond_138861
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_strided_slice_10
,lstm_tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
U
LessLessplaceholderless_lstm_strided_slice_1*
T0*
_output_shapes
: g
Less_1Lesslstm_while_loop_counterlstm_while_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         #:         #: : ::::  : : : : : : : : :	 :
 
љQ
Ц
B__inference_lstm_1_layer_call_and_return_conditional_losses_140093
inputs_0"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0M
zeros/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         #O
zeros_1/mul/yConst*
_output_shapes
: *
value	B :#*
dtype0c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
value
B :У*
dtype0_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:         #*
T0c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  #D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:         #*
T0*
Index0*
shrink_axis_maskБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#ї|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0]
mulMulSigmoid_1:y:0zeros_1:output:0*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #n
TensorArrayV2_1/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
_output_shapes
: *
valueB :
         *
dtype0T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_139993*
_num_original_outputs*
bodyR
while_body_139994*L
_output_shapes:
8: : : : :         #:         #: : : : : K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:         #*
T0^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         #M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  #h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         #e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :                  #*
T0[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*?
_input_shapes.
,:                  #:::2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : :( $
"
_user_specified_name
inputs/0
м
Д
&__inference_dense_layer_call_fn_140683

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:         #*-
_gradient_op_typePartitionedCall-138078*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_138072*
Tout
2**
config_proto

GPU 

CPU2J 8ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*.
_input_shapes
:         #::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
█P
Б
B__inference_lstm_1_layer_call_and_return_conditional_losses_137825

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0M
zeros/mul/yConst*
_output_shapes
: *
value	B :#*
dtype0_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
dtype0*
_output_shapes
: *
value
B :УY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
_output_shapes
: *
T0P
zeros/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         #O
zeros_1/mul/yConst*
value	B :#*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
_output_shapes
: *
value
B :У*
dtype0_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :#*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         #c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:
         #*
T0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
_output_shapes
: *
valueB :
         *
dtype0А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         #Б
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	#ї*
dtype0|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Д
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:ї*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
_output_shapes
: *
value	B :*
dtype0Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #]
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         #N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         #*
T0K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #n
TensorArrayV2_1/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :
*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Р
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_137724*L
_output_shapes:
8: : : : :         #:         #: : : : : *
T
2*K
output_shapes:
8: : : : :         #:         #: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_137723*
_num_original_outputsK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:         #*
T0^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:         #*
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    #   *
dtype0*
_output_shapes
:═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:
         #h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*'
_output_shapes
:         #*
T0*
Index0*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         
#[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*6
_input_shapes%
#:         
#:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : 
Ј
a
C__inference_dropout_layer_call_and_return_conditional_losses_139916

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         
#_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         
#"!

identity_1Identity_1:output:0**
_input_shapes
:         
#:& "
 
_user_specified_nameinputs
о
Е
(__inference_dense_1_layer_call_fn_140736

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-138150*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_138144*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         #*
Tin
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         #"
identityIdentity:output:0*.
_input_shapes
:         #::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
х
┌
E__inference_lstm_cell_layer_call_and_return_conditional_losses_140855

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@їj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Д
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їp
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ї*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0G
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #*
T0T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         #*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0U
mulMulSigmoid_1:y:0states_1*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         #*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         #*
T0ю
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*X
_input_shapesG
E:         @:         #:         #:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: :& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : 
В+
з
while_body_137489
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	@ї*
dtype0ј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         #*
T0Z
mulMulSigmoid_1:y:0placeholder_3*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*'
_output_shapes
:         #*
T0_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         #T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         #*
T0Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:  : : : : : : : : :	 :
 
У-
ы
#sequential_lstm_1_while_body_135838(
$sequential_lstm_1_while_loop_counter.
*sequential_lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3'
#sequential_lstm_1_strided_slice_1_0c
_tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5%
!sequential_lstm_1_strided_slice_1a
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    #   а
#TensorArrayV2Read/TensorListGetItemTensorListGetItem_tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         #Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ї*
T0Е
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	#їu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         їБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:їn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         їG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         #:         #:         #:         #T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         #V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         #Z
mulMulSigmoid_1:y:0placeholder_3*'
_output_shapes
:         #*
T0N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         #_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         #*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         #V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         #K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         #c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         #Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: g
add_3AddV2$sequential_lstm_1_while_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: «

Identity_1Identity*sequential_lstm_1_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         #"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"H
!sequential_lstm_1_strided_slice_1#sequential_lstm_1_strided_slice_1_0"└
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0*Q
_input_shapes@
>: : : : :         #:         #: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : : :	 :
 :  : : : : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*┤
serving_defaultа
E

lstm_input7
serving_default_lstm_input:0         
@;
dense_20
StatefulPartitionedCall:0         tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:Ны
└B
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+«&call_and_return_all_conditional_losses
»_default_save_signature
░__call__"Ь>
_tf_keras_sequential¤>{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "batch_input_shape": [null, 10, 64], "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 35, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "LSTM", "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 35, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 64], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "batch_input_shape": [null, 10, 64], "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 35, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "LSTM", "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 35, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 9.999999974752427e-07, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
│
regularization_losses
	variables
trainable_variables
	keras_api
+▒&call_and_return_all_conditional_losses
▓__call__"б
_tf_keras_layerѕ{"class_name": "InputLayer", "name": "lstm_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 10, 64], "config": {"batch_input_shape": [null, 10, 64], "dtype": "float32", "sparse": false, "name": "lstm_input"}}
╝

cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
+│&call_and_return_all_conditional_losses
┤__call__"Љ	
_tf_keras_layerэ{"class_name": "LSTM", "name": "lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 10, 64], "config": {"name": "lstm", "trainable": true, "batch_input_shape": [null, 10, 64], "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 35, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 64], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
Г
regularization_losses
	variables
trainable_variables
	keras_api
+х&call_and_return_all_conditional_losses
Х__call__"ю
_tf_keras_layerѓ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
њ

cell
 
state_spec
!regularization_losses
"	variables
#trainable_variables
$	keras_api
+и&call_and_return_all_conditional_losses
И__call__"у
_tf_keras_layer═{"class_name": "LSTM", "name": "lstm_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 35, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 35], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
▓
%regularization_losses
&	variables
'trainable_variables
(	keras_api
+╣&call_and_return_all_conditional_losses
║__call__"А
_tf_keras_layerЄ{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}
№

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
+╗&call_and_return_all_conditional_losses
╝__call__"╚
_tf_keras_layer«{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 35}}}}
▓
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+й&call_and_return_all_conditional_losses
Й__call__"А
_tf_keras_layerЄ{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}
з

3kernel
4bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
+┐&call_and_return_all_conditional_losses
└__call__"╠
_tf_keras_layer▓{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 35}}}}
▓
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+┴&call_and_return_all_conditional_losses
┬__call__"А
_tf_keras_layerЄ{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}
ш

=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
+├&call_and_return_all_conditional_losses
─__call__"╬
_tf_keras_layer┤{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 35}}}}
├
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_rate)mќ*mЌ3mў4mЎ=mџ>mЏHmюImЮJmъKmЪLmаMmА)vб*vБ3vц4vЦ=vд>vДHvеIvЕJvфKvФLvгMvГ"
	optimizer
 "
trackable_list_wrapper
v
H0
I1
J2
K3
L4
M5
)6
*7
38
49
=10
>11"
trackable_list_wrapper
v
H0
I1
J2
K3
L4
M5
)6
*7
38
49
=10
>11"
trackable_list_wrapper
╗
Nlayer_regularization_losses
regularization_losses
Onon_trainable_variables

Players
	variables
trainable_variables
Qmetrics
░__call__
»_default_save_signature
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
-
┼serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
Rlayer_regularization_losses
regularization_losses
Snon_trainable_variables

Tlayers
	variables
trainable_variables
Umetrics
▓__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
­

Hkernel
Irecurrent_kernel
Jbias
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
+к&call_and_return_all_conditional_losses
К__call__"│
_tf_keras_layerЎ{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
Ю
Zlayer_regularization_losses
regularization_losses
[non_trainable_variables

\layers
	variables
trainable_variables
]metrics
┤__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
^layer_regularization_losses
regularization_losses
_non_trainable_variables

`layers
	variables
trainable_variables
ametrics
Х__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
З

Kkernel
Lrecurrent_kernel
Mbias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
+╚&call_and_return_all_conditional_losses
╔__call__"и
_tf_keras_layerЮ{"class_name": "LSTMCell", "name": "lstm_cell_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_cell_1", "trainable": true, "dtype": "float32", "units": 35, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
K0
L1
M2"
trackable_list_wrapper
5
K0
L1
M2"
trackable_list_wrapper
Ю
flayer_regularization_losses
!regularization_losses
gnon_trainable_variables

hlayers
"	variables
#trainable_variables
imetrics
И__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
jlayer_regularization_losses
%regularization_losses
knon_trainable_variables

llayers
&	variables
'trainable_variables
mmetrics
║__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
:##2dense/kernel
:#2
dense/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
Ю
nlayer_regularization_losses
+regularization_losses
onon_trainable_variables

players
,	variables
-trainable_variables
qmetrics
╝__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
rlayer_regularization_losses
/regularization_losses
snon_trainable_variables

tlayers
0	variables
1trainable_variables
umetrics
Й__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 :##2dense_1/kernel
:#2dense_1/bias
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
Ю
vlayer_regularization_losses
5regularization_losses
wnon_trainable_variables

xlayers
6	variables
7trainable_variables
ymetrics
└__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
zlayer_regularization_losses
9regularization_losses
{non_trainable_variables

|layers
:	variables
;trainable_variables
}metrics
┬__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
 :#2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
Ъ
~layer_regularization_losses
?regularization_losses
non_trainable_variables
ђlayers
@	variables
Atrainable_variables
Ђmetrics
─__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
:	@ї2lstm/kernel
(:&	#ї2lstm/recurrent_kernel
:ї2	lstm/bias
 :	#ї2lstm_1/kernel
*:(	#ї2lstm_1/recurrent_kernel
:ї2lstm_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
	7

8"
trackable_list_wrapper
(
ѓ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
А
 Ѓlayer_regularization_losses
Vregularization_losses
ёnon_trainable_variables
Ёlayers
W	variables
Xtrainable_variables
єmetrics
К__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
K0
L1
M2"
trackable_list_wrapper
5
K0
L1
M2"
trackable_list_wrapper
А
 Єlayer_regularization_losses
bregularization_losses
ѕnon_trainable_variables
Ѕlayers
c	variables
dtrainable_variables
іmetrics
╔__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Б

Іtotal

їcount
Ї
_fn_kwargs
јregularization_losses
Ј	variables
љtrainable_variables
Љ	keras_api
+╩&call_and_return_all_conditional_losses
╦__call__"т
_tf_keras_layer╦{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
І0
ї1"
trackable_list_wrapper
 "
trackable_list_wrapper
ц
 њlayer_regularization_losses
јregularization_losses
Њnon_trainable_variables
ћlayers
Ј	variables
љtrainable_variables
Ћmetrics
╦__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
І0
ї1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
#:!##2Adam/dense/kernel/m
:#2Adam/dense/bias/m
%:###2Adam/dense_1/kernel/m
:#2Adam/dense_1/bias/m
%:##2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
#:!	@ї2Adam/lstm/kernel/m
-:+	#ї2Adam/lstm/recurrent_kernel/m
:ї2Adam/lstm/bias/m
%:#	#ї2Adam/lstm_1/kernel/m
/:-	#ї2Adam/lstm_1/recurrent_kernel/m
:ї2Adam/lstm_1/bias/m
#:!##2Adam/dense/kernel/v
:#2Adam/dense/bias/v
%:###2Adam/dense_1/kernel/v
:#2Adam/dense_1/bias/v
%:##2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
#:!	@ї2Adam/lstm/kernel/v
-:+	#ї2Adam/lstm/recurrent_kernel/v
:ї2Adam/lstm/bias/v
%:#	#ї2Adam/lstm_1/kernel/v
/:-	#ї2Adam/lstm_1/recurrent_kernel/v
:ї2Adam/lstm_1/bias/v
Т2с
F__inference_sequential_layer_call_and_return_conditional_losses_138234
F__inference_sequential_layer_call_and_return_conditional_losses_138794
F__inference_sequential_layer_call_and_return_conditional_losses_139153
F__inference_sequential_layer_call_and_return_conditional_losses_138261└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Т2с
!__inference__wrapped_model_135963й
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *-б*
(і%

lstm_input         
@
Щ2э
+__inference_sequential_layer_call_fn_138305
+__inference_sequential_layer_call_fn_139187
+__inference_sequential_layer_call_fn_139170
+__inference_sequential_layer_call_fn_138350└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
с2Я
@__inference_lstm_layer_call_and_return_conditional_losses_139521
@__inference_lstm_layer_call_and_return_conditional_losses_139354
@__inference_lstm_layer_call_and_return_conditional_losses_139875
@__inference_lstm_layer_call_and_return_conditional_losses_139706Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
э2З
%__inference_lstm_layer_call_fn_139529
%__inference_lstm_layer_call_fn_139537
%__inference_lstm_layer_call_fn_139883
%__inference_lstm_layer_call_fn_139891Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
─2┴
C__inference_dropout_layer_call_and_return_conditional_losses_139911
C__inference_dropout_layer_call_and_return_conditional_losses_139916┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ј2І
(__inference_dropout_layer_call_fn_139921
(__inference_dropout_layer_call_fn_139926┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
в2У
B__inference_lstm_1_layer_call_and_return_conditional_losses_140260
B__inference_lstm_1_layer_call_and_return_conditional_losses_140093
B__inference_lstm_1_layer_call_and_return_conditional_losses_140445
B__inference_lstm_1_layer_call_and_return_conditional_losses_140614Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
 2Ч
'__inference_lstm_1_layer_call_fn_140276
'__inference_lstm_1_layer_call_fn_140622
'__inference_lstm_1_layer_call_fn_140268
'__inference_lstm_1_layer_call_fn_140630Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╚2┼
E__inference_dropout_1_layer_call_and_return_conditional_losses_140655
E__inference_dropout_1_layer_call_and_return_conditional_losses_140650┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
њ2Ј
*__inference_dropout_1_layer_call_fn_140660
*__inference_dropout_1_layer_call_fn_140665┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
в2У
A__inference_dense_layer_call_and_return_conditional_losses_140676б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_dense_layer_call_fn_140683б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╚2┼
E__inference_dropout_2_layer_call_and_return_conditional_losses_140703
E__inference_dropout_2_layer_call_and_return_conditional_losses_140708┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
њ2Ј
*__inference_dropout_2_layer_call_fn_140718
*__inference_dropout_2_layer_call_fn_140713┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ь2Ж
C__inference_dense_1_layer_call_and_return_conditional_losses_140729б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense_1_layer_call_fn_140736б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╚2┼
E__inference_dropout_3_layer_call_and_return_conditional_losses_140756
E__inference_dropout_3_layer_call_and_return_conditional_losses_140761┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
њ2Ј
*__inference_dropout_3_layer_call_fn_140771
*__inference_dropout_3_layer_call_fn_140766┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ь2Ж
C__inference_dense_2_layer_call_and_return_conditional_losses_140782б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense_2_layer_call_fn_140789б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
6B4
$__inference_signature_wrapper_138373
lstm_input
м2¤
E__inference_lstm_cell_layer_call_and_return_conditional_losses_140855
E__inference_lstm_cell_layer_call_and_return_conditional_losses_140822Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ю2Ў
*__inference_lstm_cell_layer_call_fn_140883
*__inference_lstm_cell_layer_call_fn_140869Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
о2М
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_140916
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_140949Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
а2Ю
,__inference_lstm_cell_1_layer_call_fn_140977
,__inference_lstm_cell_1_layer_call_fn_140963Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 Ц
E__inference_dropout_2_layer_call_and_return_conditional_losses_140703\3б0
)б&
 і
inputs         #
p
ф "%б"
і
0         #
џ ў
+__inference_sequential_layer_call_fn_138305iHIJKLM)*34=>?б<
5б2
(і%

lstm_input         
@
p

 
ф "і         д
%__inference_lstm_layer_call_fn_139537}HIJOбL
EбB
4џ1
/і,
inputs/0                  @

 
p 

 
ф "%і"                  #Ф
C__inference_dropout_layer_call_and_return_conditional_losses_139911d7б4
-б*
$і!
inputs         
#
p
ф ")б&
і
0         
#
џ Ц
E__inference_dropout_2_layer_call_and_return_conditional_losses_140708\3б0
)б&
 і
inputs         #
p 
ф "%б"
і
0         #
џ ъ
,__inference_lstm_cell_1_layer_call_fn_140963ьKLMђб}
vбs
 і
inputs         #
KбH
"і
states/0         #
"і
states/1         #
p
ф "cб`
і
0         #
Aџ>
і
1/0         #
і
1/1         #Ф
C__inference_dropout_layer_call_and_return_conditional_losses_139916d7б4
-б*
$і!
inputs         
#
p 
ф ")б&
і
0         
#
џ ├
B__inference_lstm_1_layer_call_and_return_conditional_losses_140093}KLMOбL
EбB
4џ1
/і,
inputs/0                  #

 
p

 
ф "%б"
і
0         #
џ ╔
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_140949§KLMђб}
vбs
 і
inputs         #
KбH
"і
states/0         #
"і
states/1         #
p 
ф "sбp
iбf
і
0/0         #
EџB
і
0/1/0         #
і
0/1/1         #
џ ъ
,__inference_lstm_cell_1_layer_call_fn_140977ьKLMђб}
vбs
 і
inputs         #
KбH
"і
states/0         #
"і
states/1         #
p 
ф "cб`
і
0         #
Aџ>
і
1/0         #
і
1/1         #Б
C__inference_dense_1_layer_call_and_return_conditional_losses_140729\34/б,
%б"
 і
inputs         #
ф "%б"
і
0         #
џ ў
+__inference_sequential_layer_call_fn_138350iHIJKLM)*34=>?б<
5б2
(і%

lstm_input         
@
p 

 
ф "і         І
'__inference_lstm_1_layer_call_fn_140622`KLM?б<
5б2
$і!
inputs         
#

 
p

 
ф "і         #І
'__inference_lstm_1_layer_call_fn_140630`KLM?б<
5б2
$і!
inputs         
#

 
p 

 
ф "і         #╝
F__inference_sequential_layer_call_and_return_conditional_losses_139153rHIJKLM)*34=>;б8
1б.
$і!
inputs         
@
p 

 
ф "%б"
і
0         
џ А
A__inference_dense_layer_call_and_return_conditional_losses_140676\)*/б,
%б"
 і
inputs         #
ф "%б"
і
0         #
џ ¤
@__inference_lstm_layer_call_and_return_conditional_losses_139354іHIJOбL
EбB
4џ1
/і,
inputs/0                  @

 
p

 
ф "2б/
(і%
0                  #
џ ├
B__inference_lstm_1_layer_call_and_return_conditional_losses_140260}KLMOбL
EбB
4џ1
/і,
inputs/0                  #

 
p 

 
ф "%б"
і
0         #
џ └
F__inference_sequential_layer_call_and_return_conditional_losses_138234vHIJKLM)*34=>?б<
5б2
(і%

lstm_input         
@
p

 
ф "%б"
і
0         
џ х
@__inference_lstm_layer_call_and_return_conditional_losses_139875qHIJ?б<
5б2
$і!
inputs         
@

 
p 

 
ф ")б&
і
0         
#
џ └
F__inference_sequential_layer_call_and_return_conditional_losses_138261vHIJKLM)*34=>?б<
5б2
(і%

lstm_input         
@
p 

 
ф "%б"
і
0         
џ Ц
E__inference_dropout_1_layer_call_and_return_conditional_losses_140650\3б0
)б&
 і
inputs         #
p
ф "%б"
і
0         #
џ Ц
E__inference_dropout_1_layer_call_and_return_conditional_losses_140655\3б0
)б&
 і
inputs         #
p 
ф "%б"
і
0         #
џ Ѓ
(__inference_dropout_layer_call_fn_139921W7б4
-б*
$і!
inputs         
#
p
ф "і         
#}
*__inference_dropout_2_layer_call_fn_140713O3б0
)б&
 і
inputs         #
p
ф "і         #╝
F__inference_sequential_layer_call_and_return_conditional_losses_138794rHIJKLM)*34=>;б8
1б.
$і!
inputs         
@
p

 
ф "%б"
і
0         
џ ¤
@__inference_lstm_layer_call_and_return_conditional_losses_139521іHIJOбL
EбB
4џ1
/і,
inputs/0                  @

 
p 

 
ф "2б/
(і%
0                  #
џ }
*__inference_dropout_2_layer_call_fn_140718O3б0
)б&
 і
inputs         #
p 
ф "і         #Ѓ
(__inference_dropout_layer_call_fn_139926W7б4
-б*
$і!
inputs         
#
p 
ф "і         
#Џ
'__inference_lstm_1_layer_call_fn_140268pKLMOбL
EбB
4џ1
/і,
inputs/0                  #

 
p

 
ф "і         #К
E__inference_lstm_cell_layer_call_and_return_conditional_losses_140822§HIJђб}
vбs
 і
inputs         @
KбH
"і
states/0         #
"і
states/1         #
p
ф "sбp
iбf
і
0/0         #
EџB
і
0/1/0         #
і
0/1/1         #
џ }
*__inference_dropout_1_layer_call_fn_140660O3б0
)б&
 і
inputs         #
p
ф "і         #ю
*__inference_lstm_cell_layer_call_fn_140869ьHIJђб}
vбs
 і
inputs         @
KбH
"і
states/0         #
"і
states/1         #
p
ф "cб`
і
0         #
Aџ>
і
1/0         #
і
1/1         #Џ
'__inference_lstm_1_layer_call_fn_140276pKLMOбL
EбB
4џ1
/і,
inputs/0                  #

 
p 

 
ф "і         #▒
$__inference_signature_wrapper_138373ѕHIJKLM)*34=>EбB
б 
;ф8
6

lstm_input(і%

lstm_input         
@"1ф.
,
dense_2!і
dense_2         }
*__inference_dropout_1_layer_call_fn_140665O3б0
)б&
 і
inputs         #
p 
ф "і         #ю
*__inference_lstm_cell_layer_call_fn_140883ьHIJђб}
vбs
 і
inputs         @
KбH
"і
states/0         #
"і
states/1         #
p 
ф "cб`
і
0         #
Aџ>
і
1/0         #
і
1/1         #y
&__inference_dense_layer_call_fn_140683O)*/б,
%б"
 і
inputs         #
ф "і         #│
B__inference_lstm_1_layer_call_and_return_conditional_losses_140445mKLM?б<
5б2
$і!
inputs         
#

 
p

 
ф "%б"
і
0         #
џ }
*__inference_dropout_3_layer_call_fn_140771O3б0
)б&
 і
inputs         #
p 
ф "і         #}
*__inference_dropout_3_layer_call_fn_140766O3б0
)б&
 і
inputs         #
p
ф "і         #Ц
E__inference_dropout_3_layer_call_and_return_conditional_losses_140756\3б0
)б&
 і
inputs         #
p
ф "%б"
і
0         #
џ Ц
E__inference_dropout_3_layer_call_and_return_conditional_losses_140761\3б0
)б&
 і
inputs         #
p 
ф "%б"
і
0         #
џ {
(__inference_dense_1_layer_call_fn_140736O34/б,
%б"
 і
inputs         #
ф "і         #К
E__inference_lstm_cell_layer_call_and_return_conditional_losses_140855§HIJђб}
vбs
 і
inputs         @
KбH
"і
states/0         #
"і
states/1         #
p 
ф "sбp
iбf
і
0/0         #
EџB
і
0/1/0         #
і
0/1/1         #
џ Ї
%__inference_lstm_layer_call_fn_139883dHIJ?б<
5б2
$і!
inputs         
@

 
p

 
ф "і         
#Ї
%__inference_lstm_layer_call_fn_139891dHIJ?б<
5б2
$і!
inputs         
@

 
p 

 
ф "і         
#Ъ
!__inference__wrapped_model_135963zHIJKLM)*34=>7б4
-б*
(і%

lstm_input         
@
ф "1ф.
,
dense_2!і
dense_2         ћ
+__inference_sequential_layer_call_fn_139170eHIJKLM)*34=>;б8
1б.
$і!
inputs         
@
p

 
ф "і         Б
C__inference_dense_2_layer_call_and_return_conditional_losses_140782\=>/б,
%б"
 і
inputs         #
ф "%б"
і
0         
џ {
(__inference_dense_2_layer_call_fn_140789O=>/б,
%б"
 і
inputs         #
ф "і         ћ
+__inference_sequential_layer_call_fn_139187eHIJKLM)*34=>;б8
1б.
$і!
inputs         
@
p 

 
ф "і         х
@__inference_lstm_layer_call_and_return_conditional_losses_139706qHIJ?б<
5б2
$і!
inputs         
@

 
p

 
ф ")б&
і
0         
#
џ ╔
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_140916§KLMђб}
vбs
 і
inputs         #
KбH
"і
states/0         #
"і
states/1         #
p
ф "sбp
iбf
і
0/0         #
EџB
і
0/1/0         #
і
0/1/1         #
џ │
B__inference_lstm_1_layer_call_and_return_conditional_losses_140614mKLM?б<
5б2
$і!
inputs         
#

 
p 

 
ф "%б"
і
0         #
џ д
%__inference_lstm_layer_call_fn_139529}HIJOбL
EбB
4џ1
/і,
inputs/0                  @

 
p

 
ф "%і"                  #