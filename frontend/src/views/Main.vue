<template>
  <div class="home">
    <el-row class="outer_container">
      <el-col :span="24">
        <el-row>
          <el-col :span="14" :lg='11'>
            <Painter ref='painter'/>
          </el-col>
          <el-col :span="3">
            <el-button type="primary" :loading="loading" style="top: 40vh; position: relative" :disabled="useModel==null" @click="this.onGenerateClick">Generate!</el-button>
            <el-select v-model="useModel" placeholder="Select" size="large" style="margin-top: 20px; top: 40vh; position: relativ" @change="onChangeModel">
              <el-option
                v-for="item in options"
                :key="item.value"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
          </el-col>
          <el-col :span="7">
            <el-image v-loading="loading" :style="imageViewerStyle" :src="this.generatedImg" fit="fill" :key="this.imageKey"/>
          </el-col>
        </el-row>
      </el-col>
    </el-row>
  </div>
</template>

<script>
// @ is an alias to /src
import Painter from '@/components/Painter.vue'
import {GenerateSPADE} from '@/apis/GenerateSPADE'
import {GeneratePsP} from '@/apis/GeneratePsP'
import { ref } from 'vue'


export default {
  name: 'Main',
  components: {
    Painter
  },
  mounted() {

  },
  data() {
    return {
      generatorModel: null,
      options: [{
        value: 0,
        label: "SPADE",
      }, {
        value: 1,
        label: "PsP encoder + StyleGANv2",
      }],
      useModel: null,
      generatedImg: null,
      m1Shape: 256, // m1 : SPADE original shape
      m2Shape: 512, // m2 : PsP encoder + StyleGANv2 original shape
      imageKey: 0, // image viewer key
      imageViewerShape: null, // real shape of image viewer
      imageViewerStyle: "width: 0px; height: 0px", // default image viewer style statement
      loading : ref(false)
    }
  },
  computed() {

  },
  methods: {
    onChangeModel(val) {
      this.useModel = val
    },
    onGenerateClick() {
      this.loading = true
      let originalShape=0
      if(this.useModel==0) {
        originalShape = this.m1Shape
      } else {
        originalShape = this.m2Shape
      }
      this.imageViewerShape = Math.min(originalShape, window.innerWidth-50)
      this.imageViewerStyle = "width: " + String(this.imageViewerShape) + "px; height: " + String(this.imageViewerShape) + "px"
      let ctx = this.$refs.painter.returnCtx()
      let w = Math.min(512, window.innerWidth-50)
      let ctxArray = ctx.getImageData(0, 0, w, w)['data']
      let anno = this.canvasData2Anno(ctxArray, w)
      if(this.useModel == 0) {
        GenerateSPADE(anno).then((res)=>{
            let retv = res.data
            this.generatedImg = "data:image/jpeg;base64," + retv
            this.imageKey+=1
            this.loading = false
        })
      } else {
        GeneratePsP(anno).then((res)=>{
            let retv = res.data
            this.generatedImg = "data:image/jpeg;base64," + retv
            this.imageKey+=1
            this.loading = false
        })
      }
      
    },
    canvasData2Anno(ctxArray, w) {
      const delta = 4
      const rgbArray = []
      let row = []

      for (let i = 0; i < ctxArray.length; i = i + delta) {
        let temp = []
        if (ctxArray[i] == 208 && ctxArray[i + 1] == 208 && ctxArray[i + 2]==208) {
          temp = [0, 0, 0]
        } else {
          temp = [ctxArray[i], ctxArray[i + 1], ctxArray[i + 2]]
        }
        row.push(temp)
        if(row.length == w) {
          rgbArray.push(row)
          row = []
        }
      }

      return rgbArray
    },
  }
}






</script>


<style scoped>
/* .outer_container {
  height: 80vh;
  align-content: center;
} */
</style>