<template>
  <div class="home">
    <el-row class="outer_container">
      <el-col :span="22">
        <el-row>
          <el-col :span="13">
            <Painter ref='painter'/>
          </el-col>
          <el-col :span="3">
            <el-button type="primary" style="top: 40vh; position: relative" :disabled="useModel==null" @click="this.onGenerateClick">Generate!</el-button>
            <el-select v-model="useModel" placeholder="Select" size="large" style="margin-top: 20px; top: 40vh; position: relativ" @change="onChangeModel">
              <el-option
                v-for="item in options"
                :key="item.value"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
          </el-col>
          <el-col :span="8">
            <el-image :style="imageViewerStyle" :src="generatedImg" fit="fill" :key="imageKey"/>
          </el-col>
        </el-row>
      </el-col>
    </el-row>
    <!-- <Painter msg="Welcome to Your Vue.js App"/> -->
  </div>
</template>

<script>
// @ is an alias to /src
import Painter from '@/components/Painter.vue'

export default {
  name: 'Main',
  components: {
    Painter
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
      imageViewerStyle: "width: 100px; height: 100px", // default image viewer style statement
    }
  },
  computed() {

  },
  methods: {
    onChangeModel(val) {
      let originalShape=0
      if(val==0) {
        originalShape = this.m1Shape
      } else {
        originalShape = this.m2Shape
      }
      this.imageViewerShape = Math.min(originalShape, window.innerWidth-50)
      this.imageViewerStyle = "width: " + String(this.imageViewerShape) + "px; height: " + String(this.imageViewerShape) + "px"
      this.imageKey+=1
    },
    onGenerateClick() {
      let a = this.$refs.painter.returnCtx()
      let w = Math.min(512, window.innerWidth-50)
      console.log(a.getImageData(0, 0, w, w))
    }
  }
}






</script>


<style scoped>
/* .outer_container {
  height: 80vh;
  align-content: center;
} */
</style>