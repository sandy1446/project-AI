
var map=L.map('map',{
    crs:L.CRS.Simple
});
var sin=0;
var sout=0;
function marker(x,y){
    for(aa=0;aa<x.length;aa++){
        if(x[aa]!=null){
            if (y[aa]>=0 && y[aa]<50){
                //var layerGroup=L.marker([y[aa],-x[aa]]).addTo(map);
            }
        }
    } //console.log(s)
}
function Cnt(c,d){
    if(d.length>1){
        //console.log(c,d)
        for(j=0;j<d[0].length;j++){
            for(k=0;k<d[1].length;k++){
                r=d[0][j]
                r1=d[1][k]
                yn=r-r1
                xn=c[1][k]-c[0][j]
                //if ((r>10 && r<30) && (r1>10 && r1<30)){
                    if (yn<8 && yn>0 && xn>-1 && xn<1){
                        sin++;
                        var layerGroup=L.polyline([[d[0][j],-c[0][j]],[d[1][k],-c[1][k]]],{
                            color: 'blue',
                            weight: 1,
                        }).addTo(map);
                        //L.marker([d[0][j],-c[0][j]]).addTo(map);
                    }

                    else if(yn<0 && yn>-7 && xn>-1 && xn<1){
                        sout++;
                        var layerGroup=L.polyline([[d[0][j],-c[0][j]],[d[1][k],-c[1][k]]],{
                            color: 'red',
                            weight:1,
                        }).addTo(map);
                    }
                    document.getElementById("clickup").innerHTML=sin;
                    document.getElementById("clickdn").innerHTML=sout;
                    for(m=0;m<10000;m++){

                    }
                    
                //}
            }
        }
    }
}
var count=1;
var i=0;
var a=Array();
var b=Array();
var c=Array();
var d=Array();
var e=Array();
var name=Array();
var bounds=[[-50,-100],[100,100]];
var dataLayer = L.geoJson(dataVar);
var featureGroup=L.featureGroup();
featureGroup.addLayer(dataLayer);
featureGroup.eachLayer(function(layer){
        layer.eachLayer(function(layer,latlng){
        y=layer.feature.geometry.coordinates[0];
        x=layer.feature.geometry.coordinates[1];
        z=layer.feature.properties["Class IDs"];
        a[i]=x;
        b[i]=y;
        name[i]=z;
        if (count==1){
            time=layer.feature.properties.Time;
            //L.marker([y,x]).addTo(map)
            //console.log(time)
            i++;
        }
        
        else{
            if (layer.feature.properties.Time==time){
                //L.marker([y,x]).addTo(map);  
                for(j=0;j<a.length;j++){
                    cx=x-a[j];
                    cy=y-b[j];
                    //console.log(cx,cy)
                    if (x!=a[j] && cx>-10 && cx<10 && cy>-10 && cy<10){
                        featureGroup.removeLayer(layer.feature.properties.SN)
                        //console.log("deleted")
                        //alert(count+1+" "+x+","+j+" "+a[j])
                        a.splice(j,1)
                        b.splice(j,1)
                        //console.log(z)
                        //name.splice(j,1)
                    }
                }
                i++;
            }
            else{
                a.pop();
                b.pop();
                //name.pop();
                marker(a,b);
                c.push(a);
                d.push(b);
                //e.push(name);
                if (c.length>2){
                    c.splice(0,1);
                    d.splice(0,1);
                    //e.splice(0,1);
                    //console.log(a,b,c,d);
                }
                Cnt(c,d);
                
                i=0;
                time=layer.feature.properties.Time;
                a=[];
                b=[];
                //name=[];
                a[i]=x;
                b[i]=y;
                //name[i]=z;
                i++;
            }
        }
        count++;
});
});
L.control.mousePosition().addTo(map);
map.fitBounds(bounds);

