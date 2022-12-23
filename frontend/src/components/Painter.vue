<template>
  <div>
    <!-- mouse for PC, touch for smart phone -->
    <canvas id="sketchpad" ref="sketchpad" @mousedown="startDraw" @mousemove="drawing" @mouseup="endDraw" @touchstart="startDraw" @touchmove="drawing" @touchend="endDraw"></canvas>
    <div id="controller_container">
      <el-row id="out_row">
        <el-col :md="6" :lg="8">
          <el-radio-group v-model="penWidth">
            <el-radio label="10" border style="margin-top: 4px">
              <el-icon size='0.8vw'><EditPen /></el-icon>
            </el-radio>
            <el-radio label="20" border style="margin-top: 2px">
              <el-icon size='1.2vw'><EditPen /></el-icon>
            </el-radio>
            <el-radio label="30" border>
              <el-icon size='1.5vw'><EditPen /></el-icon>
            </el-radio>
          </el-radio-group>
        </el-col>
        <el-col :lg="13" :md="17">
          <el-radio-group v-model="penColor" :fill="penColor">
            <el-radio-button label="rgb(110,39,96)" border>brush</el-radio-button>
            <el-radio-button label="rgb(80,164,85)" border>ground</el-radio-button>
            <el-radio-button label="rgb(80,216,51)" border>water</el-radio-button>
            <el-radio-button label="rgb(60,74,191)" border>mountain</el-radio-button>
            <el-radio-button label="rgb(90,116,85)" border>sky</el-radio-button>
            <el-radio-button label="#D0D0D0" border>
              <el-icon size='0.7vw'><CloseBold /></el-icon>
            </el-radio-button>
          </el-radio-group>
        </el-col>
        <el-col :lg="2" :md="1">
          <el-button type="danger" round size='large' @click="onClearClick">
            <el-icon><Delete /></el-icon>
          </el-button>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script>
import { EditPen } from '@element-plus/icons-vue'
import { CloseBold } from '@element-plus/icons-vue'
import { Delete } from '@element-plus/icons-vue'


export default {
  name: 'Painter',
  components:{
    EditPen,
    CloseBold,
    Delete
  },
  props: {

  },
  data() {
    return {
      ctx: null,
      isDrawing: false,
      penColor: "rgb(110,39,96)",
      penWidth: "10",
      startPosX: null,
      startPosY: null,
      endPosX: null,
      endPosY: null,
      canvasWidth: null,
    }
  },
  computed: {
    
  },
  methods: {
    startDraw(e) {
      this.isDrawing = true
      this.ctx.strokeStyle = this.penColor
      this.ctx.lineWidth = this.penWidth
      this.ctx.beginPath()
      let pos = this.getMousePos(e.clientX, e.clientY)
      this.ctx.moveTo(pos.x, pos.y)
    },

    drawing(e) {
      if(this.isDrawing) {
        let pos = this.getMousePos(e.clientX, e.clientY)
        this.ctx.lineTo(pos.x, pos.y)
        this.ctx.stroke()
      }
    },

    endDraw() {
      this.isDrawing = false
    },

    getMousePos(clientX, clientY) {
      let canvasPos = this.ctx.canvas.getBoundingClientRect()
      let x = clientX-canvasPos.x
      let y = clientY-canvasPos.y

      return {x, y}
    },

    // default canvas settings
    setDefaultCanvas(){
      let canvas = this.$refs['sketchpad']
      this.canvasWidth = Math.min(512, window.innerHeight-50)
      console.log("canvasWidth : ", this.canvasWidth)
      canvas.width  = this.canvasWidth
      canvas.height  = this.canvasWidth
      let ctx =  canvas.getContext('2d')
      ctx.lineCap = "round"
      ctx.lineJoin = "round"
      // ctx.fillStyle = "#D0D0D0"
      // ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = "rgb(90,116,85)"
      ctx.fillRect(0, 0, canvas.width,Math.floor(canvas.height/2))
      ctx.fillStyle = "rgb(80,164,85)"
      ctx.fillRect(0, Math.floor(canvas.height/2), canvas.width,canvas.height)
      this.ctx = ctx
      
    },

    onClearClick() {
      // this.ctx.fillStyle = "#D0D0D0"
      // this.ctx.fillRect(0, 0, this.canvasWidth, this.canvasWidth)
      this.setDefaultCanvas()
    },

    returnCtx() {
      return this.ctx
    }

  },
  mounted() {
    this.setDefaultCanvas()
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
#sketchpad {
  border: 3px;
  border-color: black;
}

#controller_container {
  margin-top: 10px;
  width: 100%;
  height: 80px;
  background-color: white;
  border-radius: 50px;
  border: 3px solid;
  border-color: black;
  /* padding-top: 10px;
  padding-bottom: 10px; */
  /* display: inline-block;
  vertical-align: middle; */
}





#controller_container>#out_row{
  position: relative;
	top: 25%;
	margin: 0 auto;
	/* transform(translateY(- 50%)); */
}

.el-radio.is-bordered.el-radio--large {
  padding:0 10px 0 5px
}

.el-radio {
  margin-right: 0.9vw;
}


::v-deep .el-radio-button--large .el-radio-button__inner {
  padding: 13px 0.8vw 13px 0.8vw;
  font-size: 0.8vw;
}



/* h3 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
} */
</style>
