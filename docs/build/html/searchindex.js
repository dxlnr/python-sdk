Search.setIndex({docnames:["concepts","index","installation","quickstart","sdk/index","sdk/modalic","sdk/modalic.client","sdk/modalic.client.proto","sdk/modalic.client.utils","sdk/modalic.config","sdk/modalic.data","sdk/modalic.logging","sdk/modalic.server","sdk/modalic.simulation","sdk/modalic.storage","sdk/modalic.utils","sdk/modules","tutorials-and-examples/index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["concepts.rst","index.rst","installation.rst","quickstart.rst","sdk/index.rst","sdk/modalic.rst","sdk/modalic.client.rst","sdk/modalic.client.proto.rst","sdk/modalic.client.utils.rst","sdk/modalic.config.rst","sdk/modalic.data.rst","sdk/modalic.logging.rst","sdk/modalic.server.rst","sdk/modalic.simulation.rst","sdk/modalic.storage.rst","sdk/modalic.utils.rst","sdk/modules.rst","tutorials-and-examples/index.rst"],objects:{"":[[5,0,0,"-","modalic"]],"modalic.client":[[6,1,1,"","PytorchClient"],[6,1,1,"","Trainer"],[6,0,0,"-","grpc_client"],[7,0,0,"-","proto"],[6,0,0,"-","pytorch_client"],[6,0,0,"-","tensorflow_client"],[6,0,0,"-","trainer"],[8,0,0,"-","utils"]],"modalic.client.PytorchClient":[[6,2,1,"","dtype"],[6,2,1,"","loss"],[6,2,1,"","model_shape"],[6,2,1,"","round_id"],[6,3,1,"","train"]],"modalic.client.Trainer":[[6,4,1,"","model"],[6,3,1,"","train"]],"modalic.client.grpc_client":[[6,1,1,"","CommunicationLayer"],[6,1,1,"","Communicator"]],"modalic.client.grpc_client.CommunicationLayer":[[6,3,1,"","get_global_model"],[6,3,1,"","update"]],"modalic.client.grpc_client.Communicator":[[6,3,1,"","get_global_model"],[6,3,1,"","grpc_connection"],[6,3,1,"","update"]],"modalic.client.proto":[[7,0,0,"-","mosaic_pb2"],[7,0,0,"-","mosaic_pb2_grpc"]],"modalic.client.proto.mosaic_pb2_grpc":[[7,1,1,"","Communication"],[7,1,1,"","CommunicationServicer"],[7,1,1,"","CommunicationStub"],[7,5,1,"","add_CommunicationServicer_to_server"]],"modalic.client.proto.mosaic_pb2_grpc.Communication":[[7,3,1,"","GetGlobalModel"],[7,3,1,"","Update"]],"modalic.client.proto.mosaic_pb2_grpc.CommunicationServicer":[[7,3,1,"","GetGlobalModel"],[7,3,1,"","Update"]],"modalic.client.pytorch_client":[[6,1,1,"","PytorchClient"]],"modalic.client.pytorch_client.PytorchClient":[[6,2,1,"","dtype"],[6,2,1,"","loss"],[6,2,1,"","model_shape"],[6,2,1,"","round_id"],[6,3,1,"","train"]],"modalic.client.trainer":[[6,1,1,"","Trainer"]],"modalic.client.trainer.Trainer":[[6,4,1,"","model"],[6,3,1,"","train"]],"modalic.client.utils":[[8,0,0,"-","communication"],[8,0,0,"-","decor"],[8,0,0,"-","torch_utils"]],"modalic.client.utils.decor":[[8,5,1,"","train"]],"modalic.config":[[9,1,1,"","Conf"],[9,0,0,"-","config"]],"modalic.config.Conf":[[9,4,1,"","certificates"],[9,4,1,"","client_id"],[9,3,1,"","create_conf"],[9,4,1,"","data_type"],[9,3,1,"","from_toml"],[9,4,1,"","participants"],[9,4,1,"","server_address"],[9,3,1,"","set_params"],[9,4,1,"","timeout"],[9,4,1,"","training_rounds"]],"modalic.config.config":[[9,1,1,"","Conf"]],"modalic.config.config.Conf":[[9,4,1,"","certificates"],[9,4,1,"","client_id"],[9,3,1,"","create_conf"],[9,4,1,"","data_type"],[9,3,1,"","from_toml"],[9,4,1,"","participants"],[9,4,1,"","server_address"],[9,3,1,"","set_params"],[9,4,1,"","timeout"],[9,4,1,"","training_rounds"]],"modalic.data":[[10,0,0,"-","data"],[10,0,0,"-","misc"]],"modalic.data.data":[[10,1,1,"","DataPool"]],"modalic.data.misc":[[10,5,1,"","get_dataset_length"]],"modalic.logging":[[11,1,1,"","Monitor"],[11,0,0,"-","logging"],[11,0,0,"-","monitor"]],"modalic.logging.logging":[[11,1,1,"","CustomFormatter"]],"modalic.logging.logging.CustomFormatter":[[11,4,1,"","bold_red"],[11,3,1,"","custom_format"],[11,3,1,"","custom_format_str"],[11,4,1,"","faint"],[11,3,1,"","format"],[11,4,1,"","green"],[11,4,1,"","red"],[11,4,1,"","reset"],[11,4,1,"","yellow"]],"modalic.logging.monitor":[[11,1,1,"","Monitor"]],"modalic.server":[[12,0,0,"-","api"],[12,0,0,"-","server"]],"modalic.server.api":[[12,6,1,"","ServerBinaryNotFound"],[12,5,1,"","find_bin_path"]],"modalic.server.server":[[12,5,1,"","run_server"]],"modalic.simulation":[[13,1,1,"","ClientPool"],[13,0,0,"-","pool"]],"modalic.simulation.ClientPool":[[13,3,1,"","exec_single_thread"],[13,3,1,"","run"],[13,3,1,"","spawn_pool"]],"modalic.simulation.pool":[[13,1,1,"","ClientPool"]],"modalic.simulation.pool.ClientPool":[[13,3,1,"","exec_single_thread"],[13,3,1,"","run"],[13,3,1,"","spawn_pool"]],"modalic.storage":[[14,1,1,"","Storage"],[14,0,0,"-","store"]],"modalic.storage.Storage":[[14,3,1,"","upload"]],"modalic.storage.store":[[14,1,1,"","Storage"]],"modalic.storage.store.Storage":[[14,3,1,"","upload"]],"modalic.utils":[[15,0,0,"-","convert"],[15,0,0,"-","protocol"],[15,0,0,"-","serde"],[15,0,0,"-","shared"]],"modalic.utils.protocol":[[15,5,1,"","parameters_from_proto"],[15,5,1,"","parameters_to_proto"],[15,5,1,"","process_meta_to_proto"],[15,5,1,"","to_meta"]],"modalic.utils.serde":[[15,5,1,"","get_shape"],[15,5,1,"","parameters_to_weights"],[15,5,1,"","weights_to_parameters"]],"modalic.utils.shared":[[15,1,1,"","Parameters"],[15,1,1,"","ProcessMeta"]],"modalic.utils.shared.Parameters":[[15,4,1,"","data_type"],[15,4,1,"","model_version"],[15,4,1,"","tensor"]],"modalic.utils.shared.ProcessMeta":[[15,4,1,"","loss"],[15,4,1,"","round_id"]],modalic:[[6,0,0,"-","client"],[9,0,0,"-","config"],[10,0,0,"-","data"],[11,0,0,"-","logging"],[12,0,0,"-","server"],[13,0,0,"-","simulation"],[14,0,0,"-","storage"],[15,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","property","Python property"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"],"5":["py","function","Python function"],"6":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:property","3":"py:method","4":"py:attribute","5":"py:function","6":"py:exception"},terms:{"0":[4,6,8,9,10,13],"0m":11,"1":[4,6,10,13,14],"1m":11,"20m":11,"2m":11,"3":2,"31":11,"32m":11,"33":11,"5":6,"536870912":6,"8":2,"8080":[8,9],"abstract":[4,6],"byte":[6,15],"class":[4,6,7,9,10,11,13,14,15],"default":[6,9,11],"float":[6,9,15],"function":[4,6,8,13],"import":6,"int":[4,6,9,10,11,13,14,15],"return":[6,7,10,11,12,15],"static":[6,7],"true":11,"while":[8,11],If:10,The:9,abc:6,about:[1,15],abov:2,access:14,achiev:0,action:15,add_communicationservicer_to_serv:7,address:6,after:9,aggreg:[0,6,9,12],aim:1,all:[4,6],alwai:9,an:[1,6],ani:[4,6,9,10,13,14,15],api:[4,5,6,16],applic:12,ar:[0,2,13],architectur:6,arg:[6,14],associ:7,attributeerror:[4,6],avail:2,base:[6,7,9,10,11,12,13,14,15],below:17,between:6,binari:[12,15],bold_r:11,bool:6,bucket:14,buffer:7,bunch:13,call:10,call_credenti:7,can:[4,6,17],cannot:10,central:0,certif:[8,9],cfg_path:12,channel:[6,7,9],channel_credenti:7,classmethod:9,client:[0,5,9,10,11,13,16],client_id:[4,6,8,9,10],clientpool:13,code:7,collabor:0,color:11,comment:7,commun:[4,5,6,7],communicatio:6,communicationlay:6,communicationservic:7,communicationstub:[6,7],compat:[4,6],complet:6,compress:7,concept:1,concern:[4,6],concurr:13,conf:[4,6,8,9,14],config:[5,8,16],configur:[4,6,9,14],conig:9,connect:6,construct:9,contain:[4,6,9,14],content:16,context:7,convert:[5,16],coordin:0,correspond:7,create_conf:9,custom:[10,11],custom_format:11,custom_format_str:11,customformatt:11,data:[0,4,5,6,7,9,14,15,16],data_typ:[8,9,15],datapool:10,dataset:[4,6,10],datefmt:11,de:[6,9],decor:[5,6],defin:[7,9],deseri:[6,15],determin:6,devic:9,dict:[4,6,9,11,14],dictionari:14,distribut:[4,6,10],document:7,dtype:[4,6,15],dure:6,each:[0,9],edg:9,enabl:[4,6,14],encrypt:6,endpoint:[9,13],ensur:6,entiti:0,error:12,establish:[6,9],exampl:[1,4,6,8,9],except:12,exchang:0,exec_single_thread:13,execut:13,extern:9,f32:[8,9],faint:11,fals:[6,7],featur:6,feder:[1,4,6,8,9,12,13],file:[7,9,12,15],find:[12,17],find_bin_path:12,flop:17,fmt:11,focus:0,format:[11,15],formatt:11,found:12,from:[6,9],from_toml:9,futur:13,gener:7,get:[1,6,11],get_dataset_length:10,get_global_model:6,get_shap:15,getglobalmodel:7,given:9,global:[6,7],globalmeta:7,green:11,grpc:[6,9,15],grpc_client:[5,16],grpc_connect:6,ha:[4,6],handl:[7,10],handler:10,hold:[6,13],id:[4,6],identifi:[4,6,9],immedi:0,implement:6,independet:10,individu:13,inherit:6,input:[4,6],insecur:7,instal:1,instanti:14,instead:0,integ:6,intend:0,ip:6,its:15,kei:1,known:15,kwarg:6,latest:6,launch:13,layer:[4,6,14],learn:[1,4,6,8,9,12,13],len:10,length:[9,10,14],list:[6,13,15],local:[0,6,7],log:[5,6,16],log_nam:11,logback:6,logrecord:11,loop:[4,6],loss:[4,6,15],machin:0,main:13,mainli:9,manag:13,max_message_length:6,maximum:6,messag:6,meta:[7,15],metadata:[7,14],misc:[5,16],miss:7,mlop:1,modal:[1,4,17],model:[4,6,7,9,15],model_shap:[4,6],model_vers:15,modul:16,monitor:[5,16],mosaic_pb2:[5,6,15],mosaic_pb2_grpc:[5,6],msg:15,multipl:0,multipli:8,must:9,name:[11,13],ndarrai:[6,15],neg:9,nn:6,non:9,none:[4,6,7,9,11,12,13,14],notimpl:6,np:15,num_client:13,num_work:13,number:[6,9,13,17],numpi:[6,15],object:[0,4,6,7,9,10,11,13,14,15],object_nam:14,open:1,option:[4,6,7,9,10,13,14],order:6,overwrit:9,packag:16,parallel:13,param:[6,9],paramet:[4,6,9,10,11,13,15],parameters_from_proto:15,parameters_to_proto:15,parameters_to_weight:15,particip:[4,6,8,9],path:[9,12],perform:[1,8,9],period:9,pip:2,platform:[1,17],pool:[5,16],print:11,problem:0,procedur:8,process:[4,6,7,9,15],process_meta_to_proto:15,processmeta:15,produc:9,properti:[4,6],proto:[5,6],protobuf:[7,15],protocol:[5,6,7,16],provid:[0,1,6],pypi:2,python:1,pytorch:[4,6,10],pytorch_cli:[5,16],pytorchcli:[6,13],quickstart:1,rais:[4,6],raw:0,read:[1,15],record:11,red:11,refer:1,regard:[7,9],relat:12,remot:7,request:[6,7],requir:[2,9],reset:11,retri:6,root_certif:6,round:[6,9],round_id:[4,6,15],rpc:7,run:[4,6,12,13],run_serv:12,s3:14,s:0,safe:6,sampl:6,script:15,sdk:[1,2],second:9,secur:9,send:[6,7],separ:13,serd:[5,16],serial:[6,9,15],serv:[4,6],server:[0,5,6,7,9,16],server_address:[6,8,9],serverbinarynotfound:12,servermessag:7,servic:[0,7],set:[0,4,6],set_param:9,setglobalmeta:7,shape:[6,15],share:[5,16],should:9,simpl:[4,6],simul:[5,9,10,16],singl:[4,6,9,13],size:6,solv:0,some:[6,9],sourc:[1,4,6,7,8,9,10,11,12,13,14,15],spawn_pool:13,stake:6,start:1,state:9,storag:[5,16],store:[0,4,5,6,9,16],str:[6,9,11,12,13,14,15],string:[9,11],stub:6,style:11,submodul:[5,16],subpackag:16,target:7,tensor:15,tensorflow_cli:[5,16],tensorflowcli:13,thread:[6,13],threadpoolexecutor:13,threat:10,thrown:12,time:8,timeout:[7,8,9],tl:9,to_meta:15,toml:9,torch:6,torch_util:[5,6],train:[4,6,8,9,11],trainer:[4,5,16],training_round:[8,9],transfer:0,tupl:6,tutori:1,type:[6,9],under:0,underli:8,uniqu:[4,6,9],updat:[0,6,7],upload:14,us:[0,1,9,13,17],util:[5,6,16],valid:11,valu:9,version:6,via:6,wa:6,wait:9,wait_for_readi:7,want:13,weight:15,weights_to_paramet:15,when:12,where:0,which:[4,6,9,10,12,13],whole:[4,6,13],within:[4,6],worker:13,x1b:11,yellow:11,you:[13,17]},titles:["Concepts","Documentation","Installing Modalic","Quickstart","Python SDK Reference","modalic package","modalic.client package","modalic.client.proto package","modalic.client.utils package","modalic.config package","modalic.data package","modalic.logging package","modalic.server package","modalic.simulation package","modalic.storage package","modalic.utils package","modalic","Tutorials and Examples"],titleterms:{api:12,client:[4,6,7,8],commun:8,concept:0,config:9,content:[1,5,6,7,8,9,10,11,12,13,14,15],convert:15,data:10,decor:8,document:1,exampl:17,feder:0,grpc_client:6,instal:2,learn:0,log:11,misc:10,modal:[2,5,6,7,8,9,10,11,12,13,14,15,16],modul:[4,5,6,7,8,9,10,11,12,13,14,15],monitor:11,mosaic_pb2:7,mosaic_pb2_grpc:7,packag:[5,6,7,8,9,10,11,12,13,14,15],pool:13,proto:7,protocol:15,python:[2,4],pytorch_cli:6,pytorchcli:4,quickstart:3,refer:4,releas:2,sdk:4,serd:15,server:12,share:15,simul:13,stabl:2,storag:14,store:14,submodul:[6,7,8,9,10,11,12,13,14,15],subpackag:[5,6],tensorflow_cli:6,torch_util:8,trainer:6,tutori:17,util:[8,15],version:2}})