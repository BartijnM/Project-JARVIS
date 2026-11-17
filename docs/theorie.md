# Theorie van het taalmodel

Ik train Project JARVIS om het volgende token in een tekst te voorspellen. Ik begin met een bigrammodel dat één teken als context gebruikt. Daarna bouw ik een kleine causale Transformer die meerdere eerdere tekens kan gebruiken.

```drawio
<mxfile>
  <diagram id="CgMhowRAXgufTiShjw5J" name="Page-1">
    <mxGraphModel dx="1228" dy="1112" grid="1" gridSize="10" guides="1" tooltips="0" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">
      <root>
        <mxCell id="qK4Sb0BCGV25aGNJirlf-0" />
        <mxCell id="qK4Sb0BCGV25aGNJirlf-1" parent="qK4Sb0BCGV25aGNJirlf-0" />
        <UserObject label="" mermaidData="{&#xa;  &quot;data&quot;: &quot;flowchart TD\n    A[\&quot;UTF-8-tekst\&quot;] --&gt; B[\&quot;Trainings- en validatiesplit\&quot;]\n    B --&gt; C[\&quot;Gedeelde tokenizer en volgende-tokenparen\&quot;]\n    C --&gt;|invoer-ID’s| D[\&quot;J01 en J02 · Bigramtabel\&quot;]\n    C --&gt;|invoer-ID’s| E[\&quot;J03 en J04 · Token- en positie-embeddings\&quot;]\n    E --&gt; F[\&quot;Transformerblok · Causale aandacht, feed-forward, restverbindingen en normalisatie\&quot;]\n    F --&gt; G[\&quot;Uitvoerprojectie naar de woordenschat\&quot;]\n    D --&gt; H[\&quot;Logits\&quot;]\n    G --&gt; H\n    C --&gt;|doel-ID’s| I[\&quot;Cross-entropy\&quot;]\n    H --&gt; I\n    I --&gt;|alleen tijdens training| J[\&quot;Gradiënten · Handmatig in NumPy, autograd in PyTorch\&quot;]\n    J --&gt; K[\&quot;Gewichten bijwerken\&quot;]\n    H --&gt;|tijdens genereren| L[\&quot;Softmax en volgend token trekken\&quot;]\n    L --&gt; M[\&quot;Token toevoegen aan de context\&quot;]\n    M --&gt; N[\&quot;Gegenereerde ID’s decoderen\&quot;]&quot;,&#xa;  &quot;config&quot;: null&#xa;}" id="li9aAdwFS_1EtXL9e4dO-0">
          <mxCell connectable="0" parent="qK4Sb0BCGV25aGNJirlf-1" style="group;transparentBounds=1;editIcon=1;lockedGroup=0;groupPadding=10;" vertex="1">
            <mxGeometry as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="UTF-8-tekst" mermaidId="n:A" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="UTF-8-tekst" id="qK4Sb0BCGV25aGNJirlf-2">
          <mxCell parent="li9aAdwFS_1EtXL9e4dO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="144" x="310" y="10" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Trainings- en validatiesplit" mermaidId="n:B" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Trainings- en validatiesplit" id="qK4Sb0BCGV25aGNJirlf-3">
          <mxCell parent="li9aAdwFS_1EtXL9e4dO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="250" x="257" y="114" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Gedeelde tokenizer en volgende-tokenparen" mermaidId="n:C" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Gedeelde tokenizer en volgende-tokenparen" id="qK4Sb0BCGV25aGNJirlf-4">
          <mxCell parent="li9aAdwFS_1EtXL9e4dO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="379" x="193" y="218" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="J01 en J02 · Bigramtabel" mermaidId="n:D" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="J01 en J02 · Bigramtabel" id="qK4Sb0BCGV25aGNJirlf-5">
          <mxCell parent="li9aAdwFS_1EtXL9e4dO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="238" x="263" y="587" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="J03 en J04 · Token- en positie-embeddings" mermaidId="n:E" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="J03 en J04 · Token- en positie-embeddings" id="qK4Sb0BCGV25aGNJirlf-6">
          <mxCell parent="li9aAdwFS_1EtXL9e4dO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="365" x="570" y="341" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Transformerblok · Causale aandacht, feed-forward, restverbindingen en normalisatie" mermaidId="n:F" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Transformerblok · Causale aandacht, feed-forward, restverbindingen en normalisatie" id="qK4Sb0BCGV25aGNJirlf-7">
          <mxCell parent="li9aAdwFS_1EtXL9e4dO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="670" x="417" y="464" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Uitvoerprojectie naar de woordenschat" mermaidId="n:G" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Uitvoerprojectie naar de woordenschat" id="qK4Sb0BCGV25aGNJirlf-8">
          <mxCell parent="li9aAdwFS_1EtXL9e4dO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="342" x="581" y="587" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Logits" mermaidId="n:H" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Logits" id="qK4Sb0BCGV25aGNJirlf-9">
          <mxCell parent="li9aAdwFS_1EtXL9e4dO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="102" x="516" y="691" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Cross-entropy" mermaidId="n:I" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Cross-entropy" id="qK4Sb0BCGV25aGNJirlf-10">
          <mxCell parent="li9aAdwFS_1EtXL9e4dO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="159" x="159" y="814" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Gradiënten · Handmatig in NumPy, autograd in PyTorch" mermaidId="n:J" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Gradiënten · Handmatig in NumPy, autograd in PyTorch" id="qK4Sb0BCGV25aGNJirlf-11">
          <mxCell parent="li9aAdwFS_1EtXL9e4dO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="456" x="10" y="937" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Gewichten bijwerken" mermaidId="n:K" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Gewichten bijwerken" id="qK4Sb0BCGV25aGNJirlf-12">
          <mxCell parent="li9aAdwFS_1EtXL9e4dO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="213" x="132" y="1041" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Softmax en volgend token trekken" mermaidId="n:L" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Softmax en volgend token trekken" id="qK4Sb0BCGV25aGNJirlf-13">
          <mxCell parent="li9aAdwFS_1EtXL9e4dO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="306" x="549" y="814" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Token toevoegen aan de context" mermaidId="n:M" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Token toevoegen aan de context" id="qK4Sb0BCGV25aGNJirlf-14">
          <mxCell parent="li9aAdwFS_1EtXL9e4dO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="293" x="556" y="937" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="Gegenereerde ID’s decoderen" mermaidId="n:N" mermaidBaseStyle="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" mermaidBaseValue="Gegenereerde ID’s decoderen" id="qK4Sb0BCGV25aGNJirlf-15">
          <mxCell parent="li9aAdwFS_1EtXL9e4dO-0" style="html=1;whiteSpace=wrap;strokeWidth=1;fillColor=light-dark(#ECECFF,#1f2020);strokeColor=light-dark(#9370DB,#cccccc);fontColor=light-dark(#333333,#cccccc);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontSize=16;" vertex="1">
            <mxGeometry height="54" width="272" x="566" y="1041" as="geometry" />
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:A-&gt;B#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="" id="qK4Sb0BCGV25aGNJirlf-16">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-2" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="qK4Sb0BCGV25aGNJirlf-3">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:B-&gt;C#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="" id="qK4Sb0BCGV25aGNJirlf-17">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-3" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="qK4Sb0BCGV25aGNJirlf-4">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="invoer-ID’s" mermaidId="e:C-&gt;D#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="invoer-ID’s" id="qK4Sb0BCGV25aGNJirlf-18">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-4" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="qK4Sb0BCGV25aGNJirlf-5">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="invoer-ID’s" mermaidId="e:C-&gt;E#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.93;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="invoer-ID’s" id="qK4Sb0BCGV25aGNJirlf-19">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-4" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.93;exitY=1;entryX=0.5;entryY=0;" target="qK4Sb0BCGV25aGNJirlf-6">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="752" y="307" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:E-&gt;F#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="" id="qK4Sb0BCGV25aGNJirlf-20">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-6" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="qK4Sb0BCGV25aGNJirlf-7">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:F-&gt;G#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="" id="qK4Sb0BCGV25aGNJirlf-21">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-7" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="qK4Sb0BCGV25aGNJirlf-8">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:D-&gt;H#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0;entryY=0.24;" mermaidBaseValue="" id="qK4Sb0BCGV25aGNJirlf-22">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-5" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0;entryY=0.24;" target="qK4Sb0BCGV25aGNJirlf-9">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="382" y="666" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:G-&gt;H#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=1;entryY=0.24;" mermaidBaseValue="" id="qK4Sb0BCGV25aGNJirlf-23">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-8" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=1;entryY=0.24;" target="qK4Sb0BCGV25aGNJirlf-9">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="752" y="666" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="doel-ID’s" mermaidId="e:C-&gt;I#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.32;exitY=1;entryX=0.47;entryY=0;" mermaidBaseValue="doel-ID’s" id="qK4Sb0BCGV25aGNJirlf-24">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-4" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.32;exitY=1;entryX=0.47;entryY=0;" target="qK4Sb0BCGV25aGNJirlf-10">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="228" y="307" />
                <mxPoint x="228" y="368" />
                <mxPoint x="228" y="430" />
                <mxPoint x="228" y="491" />
                <mxPoint x="228" y="553" />
                <mxPoint x="228" y="614" />
                <mxPoint x="228" y="666" />
                <mxPoint x="228" y="718" />
                <mxPoint x="228" y="780" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:H-&gt;I#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0;exitY=0.72;entryX=0.71;entryY=0;" mermaidBaseValue="" id="qK4Sb0BCGV25aGNJirlf-25">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-9" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0;exitY=0.72;entryX=0.71;entryY=0;" target="qK4Sb0BCGV25aGNJirlf-10">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="315" y="780" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="alleen tijdens training" mermaidId="e:I-&gt;J#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="alleen tijdens training" id="qK4Sb0BCGV25aGNJirlf-26">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-10" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="qK4Sb0BCGV25aGNJirlf-11">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:J-&gt;K#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="" id="qK4Sb0BCGV25aGNJirlf-27">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-11" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="qK4Sb0BCGV25aGNJirlf-12">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="tijdens genereren" mermaidId="e:H-&gt;L#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=1;exitY=0.93;entryX=0.5;entryY=0;" mermaidBaseValue="tijdens genereren" id="qK4Sb0BCGV25aGNJirlf-28">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-9" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);html=1;fontSize=16;labelBackgroundColor=light-dark(#E8E8E88D,#2a2a2a8D);fontFamily=Trebuchet MS,Verdana,Arial,sans-serif;fontColor=light-dark(#333333,#cccccc);exitX=1;exitY=0.93;entryX=0.5;entryY=0;" target="qK4Sb0BCGV25aGNJirlf-13">
            <mxGeometry relative="1" as="geometry">
              <Array as="points">
                <mxPoint x="702" y="780" />
              </Array>
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:L-&gt;M#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="" id="qK4Sb0BCGV25aGNJirlf-29">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-13" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="qK4Sb0BCGV25aGNJirlf-14">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
        <UserObject label="" mermaidId="e:M-&gt;N#0" mermaidBaseStyle="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" mermaidBaseValue="" id="qK4Sb0BCGV25aGNJirlf-30">
          <mxCell edge="1" parent="li9aAdwFS_1EtXL9e4dO-0" source="qK4Sb0BCGV25aGNJirlf-14" style="curved=1;startArrow=none;endArrow=block;endSize=7;strokeColor=light-dark(#333333,#cccccc);exitX=0.5;exitY=1;entryX=0.5;entryY=0;" target="qK4Sb0BCGV25aGNJirlf-15">
            <mxGeometry relative="1" as="geometry">
              <Array as="points" />
            </mxGeometry>
          </mxCell>
        </UserObject>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>

```

Tijdens het genereren voer ik na iedere trekking de bijgewerkte context opnieuw aan het model. Het bigrammodel leest alleen het nieuwste token. De Transformer leest meerdere tokens binnen zijn contextvenster.

## J01 — Van tekens naar voorspellingen

### Tokenisatie en volgende-tokenparen

Ik splits de tekst eerst in training en validatie. Uit de trainingstekst haal ik de unieke tekens, die ik sorteer en vanaf ID `1` nummer. ID `0` reserveer ik voor onbekende tekens. Met trainingstekst `abac` wordt `a → 1`, `b → 2` en `c → 3`. Ik gebruik diezelfde koppelingen voor validatie en generatie. Bij het decoderen zie ik `�` waar een onbekend token staat.

Per split gebruik ik alle token-ID’s behalve de laatste als invoer en alle ID’s behalve de eerste als doel. Bij `abac` geeft `[1, 2, 1, 3]` dus invoer `[1, 2, 1]` en doelen `[2, 1, 3]`. Het doel op elke positie is het token één positie verder in de oorspronkelijke tekst.

### Logits, softmax en loss

Ik gebruik een tabel met vorm `(V, V)`, waarbij `V` het aantal token-ID’s is, inclusief ID `0`. De huidige ID kiest een rij. Iedere kolom geeft een score voor een mogelijk volgend token. In de code levert `weights[input_ids]` voor `B` voorbeelden een array met vorm `(B, V)`.

Softmax zet de scores, of logits, om in kansen. Ik neem per rij de exponent van elke score en deel die door de som van de exponenten. Bij logits `[0, ln(2), 0]` krijg ik kansen `[1/4, 1/2, 1/4]`. De loss voor een voorbeeld is `−ln(p)`, waarbij `p` de kans op het juiste token is en `ln` de natuurlijke logaritme. Voor doel-ID `1` krijg ik dus `−ln(1/2) ≈ 0,693`.

In `cross_entropy` trek ik eerst het maximum van elke rij af om de exponenten stabiel te berekenen. Daarna trek ik de verschoven logit van het doel af van de logaritme van de som van de exponenten. Dat geeft dezelfde loss als `−ln(p)`. Ik neem het gemiddelde over de voorbeelden.

### Gradiënten, training en generatie

In NumPy bereken ik per voorbeeld de gradiënt van de gemiddelde loss naar iedere logit. Ik begin met de softmaxkansen, trek bij de doelkolom `1` af en deel alle bijdragen door `B`. Met drie gelijke kansen en doel-ID `1` levert dat vóór het delen `[1/3, −2/3, 1/3]` op. Als twee voorbeelden dezelfde invoer-ID hebben, tel ik hun bijdragen met `np.add.at` op in dezelfde gewichtenrij. Ik werk de gewichten bij door de gradiënt maal de leersnelheid af te trekken.

Voor een controle met eindige verschillen verander ik één gewicht naar boven en naar beneden en bereken ik in beide gevallen de loss. Het verschil tussen die losses gedeeld door de afstand tussen de twee gewichtswaarden benadert de afgeleide voor dat gewicht.

Tijdens generatie zet ik de logits voor het laatste token om in kansen en trek ik daaruit een nieuw token. Ik voeg dat token toe aan de reeks en gebruik het voor de volgende voorspelling. Met dezelfde gewichten, start-ID en seed kon ik mijn NumPy-voorbeeld herhalen.

## J02 en J04 — Wat PyTorch verandert

Ik houd de tokenizer, splits, doelen en modelberekening gelijk. In het bigrammodel levert dezelfde tabelindexering met dezelfde gewichten en invoer dezelfde logits op. Cross-entropy meet in beide varianten dezelfde gemiddelde loss. Voor J02 vergelijk ik die waarden eerst op vaste invoer en gewichten.

In NumPy schrijf ik zelf uit hoe de loss via de logits afhangt van de gewichtentabel. In PyTorch werk ik met tensors. Wanneer de gewichtentensor gradiënten bijhoudt, volgt autograd de bewerkingen waarmee ik de loss bereken. Een aanroep van `loss.backward()` past de kettingregel toe tot aan de gewichten. Daarbij telt PyTorch ook bijdragen op wanneer meerdere voorbeelden dezelfde rij hebben gelezen. Dat is dezelfde optelling die ik voor het NumPy-bigrammodel met `np.add.at` uitvoer.

De gewichtenupdate blijft een aparte stap. Voor elke trainingsstap maak ik eerder opgebouwde parametergradiënten leeg, bereken ik de nieuwe loss en laat ik autograd de nieuwe gradiënten bepalen. Daarna gebruik ik die gradiënten om de gewichten bij te werken. Bij de Transformer neemt autograd ook de kettingregel door aandacht, feed-forwardlagen en normalisatie over.

## J03 — Van één teken naar een context

Ik geef de Transformer een reeks tokens. Een token-embedding geeft elke ID een vector en een positie-embedding geeft elk positie-index een vector. Ik tel die vectors op, zodat de representatie informatie bevat over zowel het teken als zijn plaats.

Causale aandacht laat elke positie informatie ophalen uit zichzelf en eerdere posities. Als de invoer `a b c` is, mag de positie van `b` aandacht geven aan `a` en `b`. De score voor aandacht naar `c` wordt gemaskeerd, zodat `b` bij het voorspellen van `c` het antwoord niet vooraf ziet. Ik controleer dit straks door een toekomstig invoertoken te wijzigen en te vergelijken of de eerdere logits gelijk blijven.

De feed-forwardlaag verwerkt de vector op iedere positie afzonderlijk. Restverbindingen tellen de invoer van een deelbewerking op bij de uitvoer. Normalisatie helpt de schaal van de activaties te beheersen. De uitvoerprojectie levert voor elke positie een score per mogelijk volgend token. Bij `B` reeksen met elk `T` posities heeft die uitvoer vorm `(B, T, V)`. Ik gebruik weer cross-entropy op de juiste volgende-token-ID’s.

## J05 — De varianten vergelijken

Ik gebruik voor elke vergelijking dezelfde tokenizer, datasplits, contextlengte en meetmethode. Ik vergelijk onder meer validatieloss, CPU-trainingsduur en gegenereerde tekst. De daling van de trainingsloss op `abab` is mijn eerste controle dat de NumPy-trainingslus leert op de gekozen voorbeelden. De validatiesplit gebruik ik om te meten hoe goed de voorspellingen werken op apart gehouden tekst.
